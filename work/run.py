"""Screen molecular datasets, run DCC-TZ, and manage single-node jobs."""
import sys
sys.dont_write_bytecode=True
import argparse
from contextlib import contextmanager
from functools import wraps
import faulthandler
import resource
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import tempfile
import time
import zipfile

# Capture the Slurm step's full mask before importing PySCF/OpenMP: binding
# narrows the Python main thread to one core, which subprocesses would inherit.
ALLOCATION_CPU_AFFINITY = sorted(os.sched_getaffinity(0))

from work.dataset import (INPUT, VALIDATION_ARCHIVE, DensityShard, atomic_json, digest,
                          input_cases, input_metadata, prepare_input)
from pyscf import dcc_tz as method

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/small-density'
RESOURCES = {
    'small': dict(threads=4, memory_gib=8, solver_memory_mb=6000),
    'standard': dict(threads=8, memory_gib=16, solver_memory_mb=12000),
}


def now(): return datetime.now(timezone.utc).isoformat()


def runtime_version():
    return dict(method=method.source_version(), workflow_sha256={
        name: digest(ROOT/name) for name in ('dataset.py', 'run.py')})


def load_config(root, verify=True):
    cfg=json.loads((Path(root)/'config.json').read_text())
    identity={k:v for k,v in cfg.items() if k not in ['run_id','created_utc']}
    expected=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
    if cfg['run_id']!=expected: raise ValueError('Run configuration was edited; use a new output directory')
    if verify:
        if runtime_version()!=cfg['runtime']: raise ValueError('Run/source or solver-default mismatch')
        if digest(cfg['input'])!=cfg['input_sha256']: raise ValueError('Input dataset checksum mismatch')
    return cfg


def prepare(args):
    meta=prepare_input(args.input, args.subset)
    root=Path(args.root).resolve()
    if (root/'config.json').exists():
        cfg=load_config(root)
        if Path(cfg['input'])!=Path(args.input).resolve(): raise ValueError('Different input dataset')
        if cfg.get('subset', 'all')!=args.subset: raise ValueError('Different input subset; use a new output directory')
    else:
        cfg=dict(format='dcc-molecular-run-v5',input=str(Path(args.input).resolve()),subset=args.subset,
                 input_sha256=digest(args.input),runtime=runtime_version(),
                 molecules=meta['count'],shards=len(meta['shards']),codec='packed64',
                 nodes_per_molecule=1,processes_per_molecule=1,
                 resources=RESOURCES,shard_classes=[s.get('resource_class','standard') for s in meta['shards']],
                 virtual_memory_gib=256,
                 job_seconds=86400,molecule_seconds=28800,commit_every=8,
                 output_free_floor_gib=100,scratch_free_floor_gib=100,
                 account='limit',partition='cpu')
        cfg['run_id']=hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
        cfg['created_utc']=now()
        atomic_json(root/'config.json',cfg)
    print(json.dumps(dict(root=str(root),molecules=cfg['molecules'],shards=cfg['shards'],
                          input_bytes=Path(args.input).stat().st_size,
                          output_free_gib=shutil.disk_usage(root).free/2**30,
                          production_submitted=False),indent=2))


def shard_resources(cfg, shard):
    return cfg['resources'][cfg['shard_classes'][shard]]


def queued_shards(root, cfg):
    """Only this run's arrays; never interpret other workloads as density jobs."""
    submitted = {}
    for path in (Path(root)/'submissions').glob('*.json'):
        record = json.loads(path.read_text())
        if record['run_id'] == cfg['run_id']:
            submitted[str(record['job_id']).split(';')[0]] = record
    if not submitted:
        return {}
    answer = subprocess.run(['squeue','--me','-r','-h','-o','%F|%K|%T|%R'],
                            capture_output=True,text=True,check=True)
    queued = {}
    for line in answer.stdout.splitlines():
        array, task, state, reason = line.split('|',3)
        if array in submitted and task.isdigit():
            shard = int(task)
            if shard in queued: raise RuntimeError('Duplicate active shard allocations')
            queued[shard] = dict(array_job_id=array, task_id=f'{array}_{task}', state=state, reason=reason)
    return queued


def shard_state(root, shard, cfg, deep=False):
    root=Path(root); path=root/'densities'/f'{shard:04d}.zip'
    status=root/'status'/f'{shard:04d}.json'
    state=json.loads(status.read_text()) if status.exists() else {}
    done=set()
    if path.exists():
        with zipfile.ZipFile(path) as z:
            index=json.loads(z.read('index.json'))
            if index['run_id']!=cfg['run_id']: raise ValueError('Shard run mismatch')
            done.update(index['entries'])
            if deep:
                for e in index['entries'].values():
                    if hashlib.sha256(z.read(e['member'])).hexdigest()!=e['sha256']:
                        raise ValueError('Archive checksum mismatch')
    cases=input_cases(shard,cfg['input'],cfg.get('subset','all'))
    if deep:
        done=DensityShard(root,shard,cfg['run_id'],cases).completed()
    names={c['name'] for c in cases}
    if not done<=names: raise ValueError('Unexpected density in shard')
    failures={n:v for n,v in state.get('failures',{}).items() if n not in done}
    return dict(shard=shard,total=len(names),committed=len(done),pending=len(names-done),
                failed=len(failures),status=state.get('status','not_started'),
                job_id=state.get('job_id'),active_case=state.get('active_case'),
                updated_utc=state.get('updated_utc'))


def report(args):
    cfg=load_config(args.root)
    queued=queued_shards(args.root,cfg)
    rows=[shard_state(args.root,i,cfg,args.verify) for i in range(cfg['shards'])]
    ids=sorted({r['job_id'] for r in rows if r['job_id']})
    accounting={}
    if ids:
        process=subprocess.run(['sacct','-n','-P','-j',','.join(ids),
                                '--format=JobIDRaw,State'],capture_output=True,text=True,check=True)
        accounting=dict(line.split('|')[:2] for line in process.stdout.splitlines() if line)
    for r in rows:
        state=accounting.get(r['job_id'],'UNKNOWN').split()[0]
        if r['status']=='running': r['status']='running' if state=='RUNNING' else state.lower()
        if r['shard'] in queued: r['queue']=queued[r['shard']]
    summary=dict(molecules=cfg['molecules'],committed=sum(r['committed'] for r in rows),
                 pending=sum(r['pending'] for r in rows),failed=sum(r['failed'] for r in rows),
                 queued_states={s:sum(r['state']==s for r in queued.values()) for s in sorted({r['state'] for r in queued.values()})},
                 active_shards=[r for r in rows if r['job_id'] or r['committed'] or r.get('queue')],
                 staged_results_note='Pending includes at most eight staged results per active shard; workers validate and recover them on restart.')
    print(json.dumps(summary,indent=2))


def timing(args):
    """Read timing statuses. Historical QM9 trials are in validation.zip."""
    paths=sorted((ROOT/'runs/timing').rglob('status/*.json'))
    if not paths:
        print('No current timing results. Stopped QM9 trials: data/validation.zip, stopped-qm9-timing-20261119/.')
        return
    rows=[]
    for path in paths:
        d=json.loads(path.read_text())
        process=subprocess.run(['sacct','-n','-P','-X','-j',str(d['job_id']),
                                '--format=JobIDRaw,JobID,State,Elapsed,NodeList'],capture_output=True,text=True,check=True)
        # The last array element can share its raw job id with the array master;
        # sacct then returns the whole array. Select this molecule's exact task.
        accounting=[line.split('|')[1:] for line in process.stdout.splitlines()
                    if line.split('|')[0]==str(d['job_id'])]
        elapsed=(datetime.now(timezone.utc)-datetime.fromisoformat(d['started_utc'])).total_seconds()
        rows.append(dict(case=d['case_name'],job=d['job_id'],accounting=accounting,
                         run=path.parent.parent.name,threads=d.get('cpus'),
                         cpu_affinity=d.get('cpu_affinity'),
                         last_stage=d.get('phase'),recorded_status=d['status'],
                         hf_stability_requested=d.get('hf_stability_requested',
                             d.get('solver_settings',{}).get('check_stability',True)),
                         elapsed_since_start_seconds=elapsed if d['status']=='running' else d.get('elapsed_seconds'),
                         completed_density=d['status']=='completed' and (path.parent.parent/'labels'/Path(d['output']).name).exists()))
    print(json.dumps(rows,indent=2))


def submit(args):
    root=Path(args.root).resolve(); cfg=load_config(root)
    if args.concurrency is not None and args.concurrency<1: raise ValueError('Concurrency must be positive')
    wanted=list(range(cfg['shards'])) if args.shards is None else [int(v) for v in args.shards.split(',')]
    if not wanted or len(set(wanted))!=len(wanted) or any(i<0 or i>=cfg['shards'] for i in wanted):
        raise ValueError('Invalid shard selection')
    queued=queued_shards(root,cfg)
    pending=[]
    for i in wanted:
        if i in queued: continue
        state=shard_state(root,i,cfg)
        if state['pending'] and (args.retry_failed or state['pending']>state['failed']): pending.append(i)
    if not pending:
        print('No pending work. Failed cases require --retry-failed.'); return
    if shutil.disk_usage(root).free<cfg['output_free_floor_gib']*2**30:
        raise RuntimeError('Output filesystem is below its free-space reserve')
    groups={name:[i for i in pending if cfg['shard_classes'][i]==name] for name in cfg['resources']}
    groups={name:shards for name,shards in groups.items() if shards}
    if args.concurrency is not None and args.concurrency<len(groups):
        raise ValueError('Concurrency must allow at least one task per resource class, or select --shards')
    script=production_script(root,args.retry_failed)
    for group_index,(name,shards) in enumerate(groups.items()):
        res=cfg['resources'][name]
        cap=(args.concurrency//len(groups)+(group_index<args.concurrency%len(groups))) if args.concurrency is not None else len(shards)
        cmd=['sbatch','--parsable',f'--account={cfg["account"]}',f'--partition={cfg["partition"]}',
             '--nodes=1','--ntasks=1',f'--cpus-per-task={res["threads"]}','--hint=nomultithread',
             f'--mem={res["memory_gib"]}G','--time=1-00:00:00',
             '--signal=B:USR1@180','--requeue','--open-mode=append','--job-name=dcc-'+name,
             '--array='+','.join(map(str,shards))+f'%{min(cap,len(shards))}',
             '--chdir='+str(ROOT.parent),'--output='+str(root/'logs/%A_%a.log')]
        print(shlex.join(cmd),flush=True)
        if not args.execute: continue
        (root/'logs').mkdir(parents=True,exist_ok=True)
        answer=subprocess.run(cmd,input=script,capture_output=True,text=True,check=True).stdout.strip()
        atomic_json(root/'submissions'/(answer.split(';')[0]+'.json'),
                    dict(utc=now(),command=cmd,script=script,job_id=answer,run_id=cfg['run_id'],
                         shards=shards,resource_class=name,resources=res))
        print('Submitted '+answer,flush=True)
    if not args.execute: print('Preview only: no jobs submitted. Add --execute to submit.')


def molecule(args):
    """Child process: one molecule, one process, no shared solver state."""
    from pyscf import lib
    from work.dataset import write_compact, read_compact, source_blocks
    import numpy as np
    cfg=load_config(args.root)
    res=shard_resources(cfg,args.shard)
    case=input_cases(args.shard,cfg['input'],cfg.get('subset','all'))[args.index]
    if int(os.environ.get('SLURM_CPUS_PER_TASK','0'))!=res['threads']:
        raise RuntimeError('Allocated CPU count differs from the shard configuration')
    lib.num_threads(res['threads'])
    def progress(stage,event): print(now(),case['name'],stage,event,flush=True)
    with tempfile.TemporaryDirectory(prefix='density-',dir=os.environ['TMPDIR']) as tmp:
        raw=Path(tmp)/'components.npz'
        with stage_reporting(progress): build_density(case,raw,res['solver_memory_mb'],cfg['run_id'])
        write_compact(raw,args.output,mode='packed64')
        _,original=source_blocks(raw); _,restored=read_compact(args.output)
        if not all(np.array_equal(a['dm'],b['dm']) and np.array_equal(a['dm_spin'],b['dm_spin'])
                   for a,b in zip(original,restored)):
            Path(args.output).unlink(missing_ok=True)
            raise ValueError('Packed64 roundtrip differs from the two original AO blocks')


def worker(args):
    root=Path(args.root).resolve(); cfg=load_config(root)
    if int(os.environ.get('SLURM_JOB_NUM_NODES','0'))!=1 or int(os.environ.get('SLURM_NTASKS','0'))!=1:
        raise RuntimeError('Worker requires one Slurm node and one task')
    shard=args.shard if args.shard is not None else int(os.environ['SLURM_ARRAY_TASK_ID'])
    res=shard_resources(cfg,shard)
    cases=input_cases(shard,cfg['input'],cfg.get('subset','all'))
    status_path=root/'status'/f'{shard:04d}.json'
    lockpath=root/'locks'/f'{shard:04d}.lock'; lockpath.parent.mkdir(parents=True,exist_ok=True)
    with lockpath.open('a') as lock:
        # Nonblocking lock prevents duplicate submissions from computing a shard twice.
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            print('Shard already owned by another worker'); return
        store=DensityShard(root,shard,cfg['run_id'],cases)
        store.commit()  # Recover completed staged labels after an interrupted job.
        state=json.loads(status_path.read_text()) if status_path.exists() else {}
        if state and state.get('run_id')!=cfg['run_id']: raise ValueError('Status run mismatch')
        failures=state.get('failures',{})
        done=store.completed()
        failures={k:v for k,v in failures.items() if k not in done}
        started=time.monotonic(); stopping=False; cancelled=False; child=None; blocked=None
        retry_failed=args.retry_failed and int(os.environ.get('SLURM_RESTART_COUNT','0'))==0
        def stop(signum,frame):
            nonlocal stopping,cancelled
            stopping=True
            if signum in (signal.SIGTERM,signal.SIGINT): cancelled=True
        old={s:signal.signal(s,stop) for s in [signal.SIGTERM,signal.SIGINT,signal.SIGUSR1]}
        def save(status,**extra):
            atomic_json(status_path,dict(run_id=cfg['run_id'],shard=shard,status=status,
                job_id=os.environ.get('SLURM_JOB_ID'),host=socket.gethostname(),updated_utc=now(),
                completed=len(done),total=len(cases),failures=failures,**extra))
        print(json.dumps(dict(shard=shard,resources=res,host=socket.gethostname(),
                              allocation_cpu_affinity=ALLOCATION_CPU_AFFINITY)),flush=True)
        try:
            for i,case in enumerate(cases):
                name=case['name']
                if name in done or (name in failures and not retry_failed): continue
                remaining=cfg['job_seconds']-(time.monotonic()-started)-300
                if stopping or remaining<60: break
                if shutil.disk_usage(root).free<cfg['output_free_floor_gib']*2**30:
                    blocked='storage_low'; break
                if shutil.disk_usage(os.environ['TMPDIR']).free<cfg['scratch_free_floor_gib']*2**30:
                    blocked='scratch_low'; break
                # This cluster schedules cores, not memory. Check actual free
                # RAM before starting another molecule on a shared node.
                available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                   if line.startswith('MemAvailable:')))*1024
                if available<(res['memory_gib']+2)*2**30:
                    blocked='memory_low'; break
                output=store.stage/(name+'.npz')
                save('running',active_case=name)
                begin=time.monotonic(); interrupted=False
                # All integrals live on the assigned node and are removed after every molecule.
                with tempfile.TemporaryDirectory(prefix=name+'-',dir=os.environ['TMPDIR']) as tmp:
                    env=dict(os.environ,TMPDIR=tmp,PYSCF_TMPDIR=tmp)
                    # Reset the inherited main-thread mask before the child's
                    # OpenMP runtime discovers its places. Stay inside srun's
                    # original allocation; never expand to the entire node.
                    cmd=['taskset','--cpu-list',','.join(map(str,ALLOCATION_CPU_AFFINITY)),
                         sys.executable,'-u','-m','work.run','molecule','--root',str(root),
                         '--shard',str(shard),'--index',str(i),'--output',str(output)]
                    child=subprocess.Popen(cmd,env=env,start_new_session=True)
                    timeout=min(cfg['molecule_seconds'],remaining)
                    while child.poll() is None:
                        if stopping or time.monotonic()-begin>timeout:
                            interrupted=stopping or remaining<cfg['molecule_seconds']
                            os.killpg(child.pid,signal.SIGTERM)
                            try: child.wait(timeout=10)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid,signal.SIGKILL); child.wait()
                            break
                        time.sleep(1)
                    code=child.returncode; child=None
                if output.exists():
                    # A child can be signalled just after atomically writing a complete label.
                    from work.dataset import label_entry
                    label_entry(output,case,cfg['run_id'])
                    done.add(name); failures.pop(name,None)
                elif not interrupted:
                    failures[name]=dict(returncode=code,seconds=time.monotonic()-begin,
                                        reason='timeout' if time.monotonic()-begin>=cfg['molecule_seconds'] else 'solver_error',
                                        job_id=os.environ.get('SLURM_JOB_ID'),utc=now())
                save('running')
                if len(list(store.stage.glob('*.npz')))>=cfg['commit_every']: store.commit()
                if interrupted: break
            store.commit()
            save('completed' if len(done)==len(cases) else 'cancelled' if cancelled else blocked or 'partial')
        except BaseException as exc:
            if child is not None and child.poll() is None:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
            save('worker_failed',error=repr(exc))
            raise
        finally:
            for s,handler in old.items(): signal.signal(s,handler)
        if cancelled: raise SystemExit(130)
        # Ask the batch wrapper to requeue only unfinished, unfailed work.
        # Solver failures remain explicit and need a deliberate --retry-failed run.
        if {c['name'] for c in cases}-done-set(failures): raise SystemExit(75)
        if failures: raise SystemExit(2)



def build_density(case, output, memory=6000, run_id=None):
    from work.dataset import write_bundle
    records = method.calculate(case, memory)
    validation = method.validate_density(case, records, memory)
    if run_id is not None:
        validation['production_run_id'] = run_id
        validation['execution'] = dict(host=socket.gethostname(),job_id=os.environ.get('SLURM_JOB_ID'),
            threads=int(os.environ.get('OMP_NUM_THREADS','1')),solver_memory_mb=memory,
            allocation_cpu_affinity=ALLOCATION_CPU_AFFINITY,
            main_thread_cpu_affinity=sorted(os.sched_getaffinity(0)))
    result = write_bundle(output, case, method.TERMS, records, validation)
    return dict(result, validation=validation)


@contextmanager
def stage_reporting(callback):
    """Observe solver stages without changing any numerical settings."""
    saved=[]
    def wrap(function,stage):
        @wraps(function)
        def run(*args,**kwargs):
            callback(stage,'start')
            value=function(*args,**kwargs)
            callback(stage,'complete')
            return value
        return run
    def replace(owner,name,function):
        saved.append((owner,name,getattr(owner,name)))
        setattr(owner,name,function)
    for name,stage in [('reference_state','scf'),('relaxed_rdm1','response_density')]:
        replace(method,name,wrap(getattr(method,name),stage))
    factory=method.cc.CCSD
    def observed_factory(*args,**kwargs):
        c=factory(*args,**kwargs)
        for name,stage in [('ao2mo','integral_transform'),('kernel','ccsd_amplitudes'),('solve_lambda','ccsd_lambda')]:
            setattr(c,name,wrap(getattr(c,name),stage))
        return c
    replace(method.cc,'CCSD',observed_factory)
    try:yield
    finally:
        for owner,name,original in reversed(saved):setattr(owner,name,original)



@contextmanager
def solver_logging():
    """Expose SCF iterations and periodic stacks without changing the solver."""
    original = method.make_mol
    def logged_mol(*args, **kwargs):
        mol = original(*args, **kwargs)
        mol.verbose = 4
        return mol
    method.make_mol = logged_mol
    faulthandler.dump_traceback_later(300, repeat=True)
    try:
        yield
    finally:
        faulthandler.cancel_dump_traceback_later()
        method.make_mol = original



def time_one(args):
    if Path(args.manifest).exists():
        manifest_bytes=Path(args.manifest).read_bytes()
    else:
        with zipfile.ZipFile(VALIDATION_ARCHIVE) as z:
            manifest_bytes=z.read('before-qm9-production/'+str(Path(args.manifest).relative_to('work')))
    case = json.loads(manifest_bytes)[args.index]
    root = Path(args.root)
    output = root / 'labels' / (case['name'] + '.npz')
    status = root / 'status' / (case['name'] + '.json')
    detail = dict(case_name=case['name'], index=args.index,
                  job_id=os.environ.get('SLURM_JOB_ID'),
                  node=socket.gethostname(),
                  cpu_affinity=sorted(os.sched_getaffinity(0)),
                  thread_environment={k:os.environ.get(k) for k in [
                      'OMP_NUM_THREADS','OMP_THREAD_LIMIT','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},
                  cpus=int(os.environ.get('SLURM_CPUS_PER_TASK', '4')),
                  solver_memory_mb=args.memory,
                  solver_settings=dict(method.PRODUCTION_SETTINGS),
                  hf_stability_requested=False,
                  started_utc=datetime.now(timezone.utc).isoformat(),
                  manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  output=str(output))
    if status.exists():
        old = json.loads(status.read_text())
        atomic_json(root / 'attempts' / (case['name'] + '.' + str(old.get('job_id')) + '.json'), old)
    start = time.perf_counter()
    events = []

    def report(state, **extra):
        usage = resource.getrusage(resource.RUSAGE_SELF)
        result = dict(detail, status=state, elapsed_seconds=time.perf_counter()-start,
                      cpu_seconds=usage.ru_utime+usage.ru_stime,
                      max_rss_kib=usage.ru_maxrss, events=events, **extra)
        atomic_json(status, result)
        return result

    def progress(stage, event):
        events.append(dict(stage=stage, event=event, elapsed_seconds=time.perf_counter()-start))
        report('running', phase=stage, phase_event=event)
        print(case['name'], stage, event, round(events[-1]['elapsed_seconds'], 3), flush=True)

    old_term = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        report('terminated', signal=signum, note='Check Slurm accounting for TIMEOUT or cancellation')
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, terminate)
    try:
        report('running')
        with stage_reporting(progress), solver_logging():
            from pyscf import lib
            lib.num_threads(detail['cpus'])
            if detail['cpus']==1 and (len(os.sched_getaffinity(0))!=1 or lib.num_threads()!=1):
                raise RuntimeError('Single-core timing requires one bound CPU and one solver thread')
            print(json.dumps(dict(case=case['name'],node=detail['node'],
                                  cpu_affinity=detail['cpu_affinity'],solver_threads=lib.num_threads(),
                                  thread_environment=detail['thread_environment'])),flush=True)
            result=build_density(case,output,args.memory)
            atomic_json(str(output)+'.json',result)
        print(json.dumps(report('completed'), indent=2), flush=True)
    except Exception as exc:
        report('failed', error=repr(exc))
        raise
    finally:
        signal.signal(signal.SIGTERM, old_term)



PRODUCTION_SCRIPT = r'''#!/bin/bash -l
# Resources and the 1-node/1-task constraint are specified by work.run submit.
set -euo pipefail
ulimit -c 0
source .local/activate.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK" OMP_THREAD_LIMIT="$SLURM_CPUS_PER_TASK" OMP_DYNAMIC=FALSE
export OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=PASSIVE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
test "${SLURM_JOB_NUM_NODES}" = 1
test "${SLURM_NTASKS}" = 1
ulimit -v $((256 * 1024 * 1024))
# Integral scratch must be node-local, never the shared output filesystem.
task_parent="${SLURM_TMPDIR:-/tmp}"
task_scratch="$(mktemp -d "$task_parent/dcc-${SLURM_JOB_ID}.XXXXXX")"
export TMPDIR="$task_scratch" PYSCF_TMPDIR="$task_scratch"
task_pid=''
task_stop() { if [[ -n "$task_pid" ]]; then kill "-$1" "$task_pid" 2>/dev/null || true; fi; }
trap 'task_stop USR1' USR1
trap 'task_stop TERM' TERM INT
trap 'rm -rf -- "$task_scratch"' EXIT
srun --nodes=1 --ntasks=1 --cpus-per-task="$SLURM_CPUS_PER_TASK" --exact --cpu-bind=cores \
    python -u -m work.run worker --root @ROOT@ @RETRY@ &
task_pid=$!
task_result=0
# wait can be interrupted by the signal trap while the worker flushes its shard.
while true; do
    wait "$task_pid" && task_result=0 || task_result=$?
    if ! kill -0 "$task_pid" 2>/dev/null; then break; fi
done
if [[ "$task_result" = 75 ]]; then
    # Successful checkpoints survive the daily allocation boundary. A manual
    # cancellation exits 130 and never requeues itself.
    sleep 30
    scontrol requeue "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
    exit 0
fi
exit "$task_result"
'''


def production_script(root, retry_failed=False):
    return PRODUCTION_SCRIPT.replace('@ROOT@', shlex.quote(str(root))).replace(
        '@RETRY@', '--retry-failed' if retry_failed else '')


def build(args):
    from pyscf import lib
    from work.dataset import read_cases
    lib.num_threads(args.threads)
    case = read_cases(args.manifest)[args.index]
    result = build_density(case, args.output, args.memory)
    atomic_json(str(args.output)+'.json', result)
    print(json.dumps(result, indent=2))


def inspect_ixc2(args):
    from work.dataset import screen_ixc2
    report=screen_ixc2(args.training_run,args.output,args.max_heavy,args.max_nao)
    print(json.dumps(dict(output=args.output,train_population=report['train_population'],
                         size_sweep=report['size_sweep'],selected=report['selected'],
                         expanded_dispersion=report['expanded_dispersion'],
                         exported_cases=report['exported_cases']),indent=2))


def submit_timing(args):
    """Generate a single-node timing job; never submit without --execute."""
    root = Path(args.root).resolve()
    cmd = ['sbatch', '--parsable', '--account=limit', '--partition=cpu',
           '--nodes=1', '--ntasks=1', f'--cpus-per-task={args.threads}',
           '--hint=nomultithread', '--mem=96G', '--time=3-00:00:00',
           '--job-name=dcc-timing', '--chdir='+str(ROOT.parent),
           '--output='+str(root/'logs/%j.log')]
    if args.threads == 52:
        cmd.append('--exclusive')
    if args.node:
        cmd.append('--nodelist='+args.node)
    command = shlex.join(['python', '-u', '-m', 'work.run', 'time-one',
                         '--manifest', args.manifest, '--index', str(args.index),
                         '--root', str(root), '--memory', '60000'])
    if args.threads == 1:
        command = 'srun --nodes=1 --ntasks=1 --cpus-per-task=1 --exact --cpu-bind=threads '+command
    script = '''#!/bin/bash -l
set -euo pipefail
source .local/activate.sh
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=@THREADS@ OMP_THREAD_LIMIT=@THREADS@ OMP_DYNAMIC=FALSE
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export OMP_PLACES=cores OMP_PROC_BIND=spread OMP_WAIT_POLICY=PASSIVE
test "$SLURM_JOB_NUM_NODES" = 1
test "$SLURM_NTASKS" = 1
ulimit -c 0
ulimit -v $((256 * 1024 * 1024))
task_scratch=$(mktemp -d "${SLURM_TMPDIR:-/tmp}/dcc-timing-${SLURM_JOB_ID}.XXXXXX")
export TMPDIR="$task_scratch" PYSCF_TMPDIR="$task_scratch"
trap 'rm -rf -- "$task_scratch"' EXIT
python - <<'PY'
from pathlib import Path
free=int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:')))
if free < 104*1024**2:
    raise SystemExit('Less than 96 GiB available RAM plus 8 GiB reserve')
PY
@COMMAND@
'''.replace('@THREADS@', str(args.threads)).replace('@COMMAND@', command)
    print(shlex.join(cmd), flush=True)
    if not args.execute:
        print('Preview only: no jobs submitted. Add --execute to submit.')
        return
    (root/'logs').mkdir(parents=True, exist_ok=True)
    answer = subprocess.run(cmd, input=script, text=True, capture_output=True, check=True).stdout.strip()
    atomic_json(root/'submissions'/(answer.split(';')[0]+'.json'),
                dict(utc=now(), command=cmd, script=script, job_id=answer,
                     runtime=runtime_version()))
    print('Submitted '+answer)






def prepare_small(args):
    from work.dataset import prepare_small as assemble
    result=assemble(args.output, sources=args.sources, base=args.base)
    print(json.dumps({k:result[k] for k in (
        'production_count','closed_shell_count','open_shell_count','base_ixc2_count','added_source_dimers','added_source_monomers',
        'exact_duplicates_removed','open_shell_candidates','derived_bond_challenge','nao_tz')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--input',default=str(INPUT))
    a.add_argument('--subset',choices=['all','closed-shell','open-shell'],default='all')
    b=sub.add_parser('submit'); b.add_argument('--execute',action='store_true')
    b.add_argument('--concurrency',type=int,help='Optional array-task cap; default lets Slurm fill available CPU quota'); b.add_argument('--shards')
    b.add_argument('--retry-failed',action='store_true')
    c=sub.add_parser('report'); c.add_argument('--verify',action='store_true')
    sub.add_parser('timing')
    d=sub.add_parser('worker'); d.add_argument('--shard',type=int)
    d.add_argument('--retry-failed',action='store_true')
    e=sub.add_parser('molecule'); e.add_argument('--shard',type=int,required=True)
    e.add_argument('--index',type=int,required=True); e.add_argument('--output',required=True)
    for parser in [a,b,c,d,e]: parser.add_argument('--root',default=str(RUN))
    f=sub.add_parser('time-one'); f.add_argument('--manifest',required=True)
    f.add_argument('--index',type=int,required=True); f.add_argument('--root',required=True)
    f.add_argument('--memory',type=int,default=60000)
    g=sub.add_parser('build'); g.add_argument('--manifest',required=True)
    g.add_argument('--index',type=int,required=True); g.add_argument('--output',required=True)
    g.add_argument('--memory',type=int,default=6000); g.add_argument('--threads',type=int,default=1)
    h=sub.add_parser('submit-timing'); h.add_argument('--index',type=int,required=True)
    h.add_argument('--manifest',required=True)
    h.add_argument('--threads',type=int,choices=[1,52],required=True)
    h.add_argument('--root',required=True); h.add_argument('--node')
    h.add_argument('--execute',action='store_true')
    i=sub.add_parser('inspect-ixc2'); i.add_argument('--training-run',required=True)
    i.add_argument('--output',default=str(ROOT/'data/ixc2-small.zip'))
    i.add_argument('--max-heavy',type=int,default=3)
    i.add_argument('--max-nao',type=int,default=200)
    j=sub.add_parser('prepare-small',help='Assemble pinned public geometries; no quantum calculations')
    j.add_argument('--output',default=str(INPUT))
    j.add_argument('--sources',help='Optional directory of already downloaded, checksum-pinned sources')
    j.add_argument('--base',default=str(ROOT/'data/ixc2-small.zip'))
    args=p.parse_args(); globals()[args.command.replace('-', '_')](args)


if __name__=='__main__': main()
