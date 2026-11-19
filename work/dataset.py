"""Molecular inputs, lossless DCC-TZ density storage, and restartable ZIP shards.

Each closed-shell label stores two signed symmetric float64 AO matrices:
the aTZ MP2 base and the aDZ (CCSD - MP2) correction. The matrices keep their
own basis definitions; real-space density is evaluated when requested.
"""
import bz2
import ast
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import urllib.request
import zipfile

import numpy as np
from pyscf import gto
from pyscf.dcc_tz import density_diagnostics, evaluate_density

INPUT = Path(__file__).resolve().parent/'data/small-density.zip'
VALIDATION_ARCHIVE = Path(__file__).resolve().parent/'data/validation.zip'
SOURCE_URL = 'https://qmml.org/Datasets/gdb9-14b.zip'
SOURCE_SHA256 = '1261ac7dbeb0a899246f191c9467c23423111562c8fbe88b2f6ba6e6111a5e7a'
FORMAT = 'dcc-tz-closed-shell-v1'
METHOD = 'DCC-TZ'

def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)



def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()



def case_digest(case):
    return hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':')).encode()).hexdigest()



def input_metadata(path=INPUT):
    with zipfile.ZipFile(path) as z:
        return json.loads(z.read('dcc/metadata.json'))



def input_cases(shard, path=INPUT):
    with zipfile.ZipFile(path) as z:
        meta = json.loads(z.read('dcc/metadata.json'))
        spec = meta['shards'][shard]
        raw = z.read(spec['member'])
        if hashlib.sha256(raw).hexdigest() != spec['sha256']:
            raise ValueError('Input shard checksum mismatch')
        return [json.loads(s) for s in raw.splitlines()]



def read_cases(manifest):
    """Read a JSON case list, including archive.zip::member.json."""
    filename, separator, member = str(manifest).partition('::')
    if separator:
        with zipfile.ZipFile(filename) as z:
            raw = z.read(member)
    else:
        raw = Path(filename).read_bytes()
    cases = json.loads(raw)
    if not isinstance(cases,list):
        raise ValueError('Expected a JSON list of molecular cases')
    return cases


def prepare_input(path=INPUT):
    """Validate normalized shards, or normalize an original QM9 source archive."""
    path = Path(path)
    with zipfile.ZipFile(path) as z:
        if 'dcc/metadata.json' in z.namelist():
            meta = input_metadata(path)
            for i in range(len(meta['shards'])): input_cases(i, path)
            return meta
    if digest(path) != SOURCE_SHA256:
        raise ValueError('Unexpected original QM9 archive checksum')
    with zipfile.ZipFile(path) as z:
        excluded_text = z.read('uncharacterized.txt').decode()
        excluded = {int(m.group(1)) for line in excluded_text.splitlines()
                    if (m := re.match(r'^\s*(\d+)\s+\S+', line))}
        if len(excluded) != 3054:
            raise ValueError(f'Unexpected excluded count: {len(excluded)}')
        cases, seen = [], set()
        with z.open('dsgdb9nsd.xyz.bz2') as packed:
            with io.TextIOWrapper(bz2.BZ2File(packed)) as f:
                for line in f:
                    if not line.strip(): continue
                    n = int(line)
                    properties = f.readline().split()
                    if properties[0] != 'gdb': raise ValueError('Unexpected XYZ header')
                    ident = int(properties[1])
                    if ident in seen: raise ValueError('Duplicate QM9 id')
                    seen.add(ident)
                    atoms = []
                    for _ in range(n):
                        row = f.readline().split()
                        xyz = [float(v.replace('*^','e').replace('D','E')) for v in row[1:4]]
                        if row[0] not in {'H','C','N','O','F'} or len(xyz)!=3 or not all(map(math.isfinite,xyz)):
                            raise ValueError('Invalid QM9 atom')
                        atoms.append([row[0], xyz])
                    f.readline()  # frequencies
                    smiles = f.readline().split()
                    inchi = f.readline().split()
                    if not inchi[0].startswith('InChI='): raise ValueError('Truncated QM9 entry')
                    if ident in excluded: continue
                    electrons = sum({'H':1,'C':6,'N':7,'O':8,'F':9}[a] for a,_ in atoms)
                    if electrons % 2: raise ValueError('Odd-electron neutral QM9 entry')
                    cases.append(dict(name=f'qm9_{ident:06d}',qm9_id=ident,atom=atoms,
                                      unit='Angstrom',charge=0,spin=0,smiles=smiles[0],
                                      source='QM9 (QM/ML corrected XYZ archive)'))
    if seen != set(range(1,133886)) or len(cases)!=130831 or not excluded <= seen:
        raise ValueError('QM9 completeness check failed')
    cases.sort(key=lambda c:c['qm9_id'])
    # Build a new file, then replace; an interrupted preparation keeps the source intact.
    tmp = path.with_suffix('.prepare.tmp')
    try:
        specs=[]
        with zipfile.ZipFile(path) as src, zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as out:
            for info in src.infolist(): out.writestr(info,src.read(info.filename))
            for i,start in enumerate(range(0,len(cases),1024)):
                chunk=cases[start:start+1024]
                raw=('\n'.join(json.dumps(c,separators=(',',':')) for c in chunk)+'\n').encode()
                member=f'dcc/inputs/{i:04d}.jsonl'
                out.writestr(member,raw)
                specs.append(dict(member=member,count=len(chunk),sha256=hashlib.sha256(raw).hexdigest(),
                                  first_id=chunk[0]['qm9_id'],last_id=chunk[-1]['qm9_id']))
            meta=dict(format='dcc-qm9-input-v1',source_url=SOURCE_URL,source_sha256=SOURCE_SHA256,
                      original=133885,excluded=3054,count=len(cases),shard_size=1024,shards=specs,
                      selection='All original geometries except the published uncharacterized list; no geometry optimization.',
                      density_labels_included=False)
            out.writestr('dcc/metadata.json',json.dumps(meta,indent=2))
        with zipfile.ZipFile(tmp) as check:
            if check.testzip() is not None: raise ValueError('Prepared input CRC failure')
        with tmp.open('rb') as f: os.fsync(f.fileno())
        tmp.replace(path)
    finally: tmp.unlink(missing_ok=True)
    return meta



def write_bundle(path, case, terms, records, validation):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays, components = {}, []
    for i, (coefficient, role, method) in enumerate(terms):
        record = records[role][method]
        key = f"component_{i}"
        arrays[key] = record["dm"]
        spin_key = None
        if record.get("dm_spin") is not None:
            spin_key = key+"_spin"
            arrays[spin_key] = record["dm_spin"]
            if arrays[spin_key].shape != (2,*record["dm"].shape) or not np.allclose(
                    arrays[spin_key].sum(axis=0),record["dm"],atol=1e-10,rtol=0):
                raise ValueError("Spin components do not sum to total density")
        density_diagnostics(gto.loads(record["meta"]["mol"]), record["dm"])
        components.append(dict(array=key, spin_array=spin_key, coefficient=coefficient, role=role,
                               method=method, meta=record["meta"]))
    if sum(x[0] for x in terms) != 1:
        raise ValueError("Density coefficients must sum to one")
    metadata = dict(format="densitylab-components-v1", case=case,
                    coordinate_unit="Bohr", density_unit="electron/Bohr^3",
                    field="electron number density; not multiplied by electron charge; no nuclear delta functions",
                    label_status="research_candidate", validated_near_cbs=False,
                    components=components, validation=validation)
    arrays["metadata"] = np.array(json.dumps(metadata, allow_nan=False))
    tmp = path.with_name(path.name+f".{os.getpid()}.tmp")
    with tmp.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    tmp.replace(path)
    return dict(file=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                case_name=case["name"], validated_near_cbs=False,
                bytes=path.stat().st_size, n_components=len(components))



def read_bundle(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data["metadata"]))
        if metadata["format"] != "densitylab-components-v1":
            raise ValueError("Unsupported density format")
        records = [(c["coefficient"], dict(meta=c["meta"], dm=data[c["array"]].copy(),
                    dm_spin=data[c["spin_array"]].copy() if c.get("spin_array") else None))
                   for c in metadata["components"]]
    return metadata, records



def read_archived_bundle(member):
    """Read an original NPZ losslessly from the consolidated validation archive."""
    with zipfile.ZipFile(VALIDATION_ARCHIVE) as archive:
        return read_bundle(io.BytesIO(archive.read(member)))



def _symmetric(a):
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1] or not np.isfinite(a).all():
        raise ValueError('Expected a finite real square density matrix')
    if np.max(np.abs(a-a.T)) > 1e-9:
        raise ValueError('Density is not symmetric')
    return (a+a.T)*0.5



def _basis_equal(a, b):
    """Compare actual AO definitions, not incidental Mole memory/log settings."""
    return (a.cart == b.cart and a.charge == b.charge and a.spin == b.spin
            and a._atom == b._atom and a._basis == b._basis
            and a._ecp == b._ecp and a._pseudo == b._pseudo)



def source_blocks(path):
    metadata, records = read_bundle(path)
    if metadata['case'].get('spin', 0) != 0:
        raise ValueError('This codec is for restricted closed-shell labels only')
    specs = [(c['coefficient'], c['role'], c['method']) for c in metadata['components']]
    if specs != [(1, 'T', 'mp2'), (1, 'A', 'ccsd'), (-1, 'A', 'mp2')]:
        raise ValueError('Expected the three frozen H3C components in order')
    for _, r in records:
        if r['meta'].get('reference') != 'RHF':
            raise ValueError('Only restricted references can omit spin matrices')
        if r.get('dm_spin') is None or not np.allclose(
                r['dm_spin'], r['dm'][None]*0.5, rtol=0, atol=1e-12):
            raise ValueError('Source does not contain equal alpha and beta densities')
    mols = [gto.loads(r['meta']['mol']) for _, r in records]
    if not _basis_equal(mols[1], mols[2]):
        raise ValueError('aDZ subtraction requires identical AO definitions')
    if [r['meta']['basis'].lower() for _, r in records] != [
            'aug-cc-pvtz', 'aug-cc-pvdz', 'aug-cc-pvdz']:
        raise ValueError('Unexpected basis')
    d = [r['dm'] for _, r in records]
    blocks = [dict(name='base_aTZ', mol=mols[0], dm=_symmetric(d[0])),
              dict(name='delta_aDZ', mol=mols[1], dm=_symmetric(d[1]-d[2]))]
    return metadata, blocks



def decode_block(spec, arrays, mol):
    """Decode exact triangular packing; experimental lossy codecs are archived."""
    if spec['codec'] != 'packed64':
        raise ValueError('Only the production packed64 codec is supported')
    n = mol.nao_nr()
    triangle = np.asarray(arrays['triangle'])
    if (triangle.dtype != np.float64 or triangle.shape != (n*(n+1)//2,)
            or not np.isfinite(triangle).all()):
        raise ValueError('Invalid packed AO matrix')
    dm = np.zeros((n,n))
    i, j = np.tril_indices(n)
    dm[i,j] = dm[j,i] = triangle
    return dm


def encode_block(dm, mol, mode='packed64', tolerance=5e-7):
    if mode != 'packed64':
        raise ValueError('Production uses lossless packed64')
    dm = _symmetric(dm)
    arrays = dict(triangle=dm[np.tril_indices(len(dm))].copy())
    spec = dict(codec='packed64', requested_codec=mode, nao=len(dm))
    restored = decode_block(spec, arrays, mol)
    if not np.array_equal(dm, restored):
        raise ValueError('Packed64 roundtrip changed the density')
    count = float(np.einsum('ij,ji', dm, mol.intor_symmetric('int1e_ovlp')))
    # Exact packing requires no overlap inversion, even for redundant AO bases.
    spec.update(l1_bound_electrons=0., electron_count_before=count,
                electron_count_after=count, array_bytes=arrays['triangle'].nbytes)
    return spec, arrays


def write_compact(source, output, mode='packed64', tolerance=1e-6):
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve():
        raise ValueError('Never overwrite a source density bundle')
    metadata, blocks = source_blocks(source)
    arrays, specs = {}, []
    for i, block in enumerate(blocks):
        spec, data = encode_block(block['dm'], block['mol'], mode, tolerance/2)
        spec.update(name=block['name'], mol=block['mol'].dumps(),
                    arrays={k: f'block_{i}_{k}' for k in data})
        specs.append(spec)
        arrays.update({spec['arrays'][k]: v for k, v in data.items()})
    # Retain original case, basis conventions, method settings and validation.
    meta = dict(format=FORMAT, method=METHOD, protocol_alias='H3C',
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_metadata=metadata, blocks=specs,
                coordinate_unit='Bohr', density_unit='electron/Bohr^3',
                spin_representation='restricted: alpha=beta=total/2',
                requested_l1_bound_electrons=tolerance,
                l1_bound_electrons=sum(s['l1_bound_electrons'] for s in specs),
                bound_note='Analytical L1 inequality, evaluated numerically in float64; no certified floating-point rounding bound.')
    arrays['metadata'] = np.array(json.dumps(meta, allow_nan=False))
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name+f'.{os.getpid()}.tmp')
    try:
        with tmp.open('wb') as handle:
            np.savez_compressed(handle, **arrays)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return dict(file=str(output), sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                bytes=output.stat().st_size, source_bytes=source.stat().st_size,
                array_bytes=sum(s['array_bytes'] for s in specs),
                l1_bound_electrons=meta['l1_bound_electrons'],
                codecs=[s['codec'] for s in specs], blocks=specs)



def read_compact(path):
    with np.load(path, allow_pickle=False) as data:
        metadata = json.loads(str(data['metadata']))
        if metadata.get('format') != FORMAT:
            raise ValueError('Unsupported compact density format')
        records = []
        for spec in metadata['blocks']:
            mol = gto.loads(spec['mol'])
            dm = decode_block(spec, {k: data[v] for k, v in spec['arrays'].items()}, mol)
            records.append(dict(meta=dict(mol=spec['mol']), dm=dm, dm_spin=np.array([dm/2, dm/2])))
    return metadata, records



def evaluate_compact(path, coords, spin=False):
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or not np.isfinite(coords).all():
        raise ValueError('Expected finite (n,3) coordinates in Bohr')
    _, records = read_compact(path)
    rho = sum(evaluate_density(r, coords) for r in records)
    return np.array([rho/2, rho/2]) if spin else rho



def label_entry(path, case, run_id):
    """Validate provenance before admitting a computed density to a shard."""
    import numpy as np
    with np.load(path,allow_pickle=False) as label:
        meta=json.loads(str(label['metadata']))
        if meta.get('format')!='dcc-tz-closed-shell-v1': raise ValueError('Wrong density format')
        source=meta['source_metadata']
        if source['case']!=case: raise ValueError('Density geometry/state mismatch')
        if not source['validation'].get('quality_pass'): raise ValueError('Unvalidated numerical label')
        if source['validation'].get('production_run_id')!=run_id: raise ValueError('Density run identity mismatch')
        if any(b['codec']!='packed64' for b in meta['blocks']): raise ValueError('Production requires packed64')
        if len(meta['blocks'])!=2: raise ValueError('Expected two AO blocks')
        for b in meta['blocks']:
            a=label[b['arrays']['triangle']]
            if a.dtype!=np.float64 or a.shape!=(b['nao']*(b['nao']+1)//2,) or not np.isfinite(a).all():
                raise ValueError('Invalid packed density')
    return dict(member=case['name']+'.npz',sha256=digest(path),bytes=Path(path).stat().st_size,
                case_sha256=case_digest(case))



class DensityShard:
    """Single-writer shard. Caller holds a per-shard flock for its lifetime.

    Individual results are atomic staged files. A merge writes a fresh ZIP,
    verifies it, fsyncs and replaces the old ZIP before deleting staged files.
    Thus interruption never modifies the previously committed archive.
    """
    def __init__(self, root, shard, run_id, cases):
        self.root=Path(root)
        self.path=self.root/'densities'/f'{shard:04d}.zip'
        self.stage=self.root/'staging'/f'{shard:04d}'
        self.run_id=run_id
        self.cases={c['name']:c for c in cases}
        self.entries={}
        if self.path.exists():
            with zipfile.ZipFile(self.path) as z:
                index=json.loads(z.read('index.json'))
                if index['run_id']!=run_id: raise ValueError('Shard belongs to another run')
                self.entries=index['entries']
                for name,e in self.entries.items():
                    if name not in self.cases or e['case_sha256']!=case_digest(self.cases[name]):
                        raise ValueError('Shard/input identity mismatch')
                    if hashlib.sha256(z.read(e['member'])).hexdigest()!=e['sha256']:
                        raise ValueError('Committed density checksum mismatch')
        self.stage.mkdir(parents=True,exist_ok=True)

    def completed(self):
        names=set(self.entries)
        for p in self.stage.glob('*.npz'):
            if p.stem not in self.cases: raise ValueError('Unexpected staged case')
            label_entry(p,self.cases[p.stem],self.run_id)
            names.add(p.stem)
        return names

    def commit(self):
        staged=sorted(self.stage.glob('*.npz'))
        if not staged: return
        entries=dict(self.entries)
        for p in staged:
            e=label_entry(p,self.cases[p.stem],self.run_id)
            if p.stem in entries and entries[p.stem]['sha256']!=e['sha256']:
                raise ValueError('Conflicting completed density; refusing overwrite')
            entries[p.stem]=e
        self.path.parent.mkdir(parents=True,exist_ok=True)
        tmp=self.path.with_suffix('.zip.tmp')
        try:
            with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_STORED,allowZip64=True) as out:
                if self.path.exists():
                    with zipfile.ZipFile(self.path) as old:
                        for name,e in self.entries.items(): out.writestr(e['member'],old.read(e['member']))
                for p in staged:
                    if p.stem not in self.entries: out.write(p,entries[p.stem]['member'])
                out.writestr('index.json',json.dumps(dict(format='dcc-density-shard-v1',run_id=self.run_id,entries=entries)))
            with zipfile.ZipFile(tmp) as check:
                for e in entries.values():
                    if hashlib.sha256(check.read(e['member'])).hexdigest()!=e['sha256']:
                        raise ValueError('New shard checksum mismatch')
            with tmp.open('rb') as f: os.fsync(f.fileno())
            tmp.replace(self.path)
            fd=os.open(self.path.parent,os.O_DIRECTORY)
            try: os.fsync(fd)
            finally: os.close(fd)
            self.entries=entries
            for p in staged: p.unlink()
        finally: tmp.unlink(missing_ok=True)



def read_density(root, shard, name):
    """Return (metadata, two AO blocks) without extracting a ZIP shard."""
    with zipfile.ZipFile(Path(root)/'densities'/f'{shard:04d}.zip') as z:
        index=json.loads(z.read('index.json')); entry=index['entries'][name]
        raw=z.read(entry['member'])
        if hashlib.sha256(raw).hexdigest()!=entry['sha256']: raise ValueError('Density checksum mismatch')
    return read_compact(io.BytesIO(raw))


def screen_ixc2(run_root, output, max_heavy=3, max_nao=200):
    """Screen the exact SQLite pinned by an IXC2 run; never alter its data.

    Select complete energy observations whose constituent states all satisfy
    the size limits. Count unique state IDs, not energy labels or repeated
    optimizer visits. Keep all ten distances of each retained D442 curve.
    Atomic references are counted separately from molecular density cases.
    """
    if max_heavy < 1 or max_nao < 1:
        raise ValueError('Size limits must be positive')
    run_root = Path(run_root).resolve()
    config = json.loads((run_root/'run.json').read_text())
    db_path = Path(config['data'])
    if not db_path.is_absolute():
        db_path = Path(config['repository'])/db_path
    db_path = db_path.resolve()
    checksum = digest(db_path)
    if checksum != config['data_sha256']:
        raise ValueError('Database differs from the one pinned by the IXC2 run')
    db = sqlite3.connect(db_path.as_uri()+'?mode=ro', uri=True)
    try:
        metadata = {k:json.loads(v) for k,v in db.execute('select key,value from metadata')}
        if metadata['schema'] != 'ixc2-stage1-relative-v2' or not metadata['ready']:
            raise ValueError('Unsupported or incomplete IXC2 database')
        states = {i:json.loads(raw) for i,raw in db.execute('select id,data from states')}
        labels = [json.loads(raw) for raw, in db.execute('select data from labels')]
    finally:
        db.close()
    elements = sorted({e for s in states.values() for e in s['symbols']})
    basis_counts = {basis:{e:sum((2*b[0]+1)*(len(b[1])-1) for b in gto.basis.load(basis,e))
                          for e in elements} for basis in ('aug-cc-pvdz','aug-cc-pvtz')}
    sizes = {i:dict(natom=len(s['symbols']), heavy=sum(e!='H' for e in s['symbols']),
                    nao_dz=sum(basis_counts['aug-cc-pvdz'][e] for e in s['symbols']),
                    nao_tz=sum(basis_counts['aug-cc-pvtz'][e] for e in s['symbols']))
             for i,s in states.items()}
    train = [l for l in labels if l['split']=='train']
    heldout = {i for l in labels if l['split']!='train' for i in l['terms']}
    population = dict(Counter(l['category'] for l in train))
    if population != config['sampling']['population']:
        raise ValueError('Training population differs from the active run configuration')
    categories = ('atomization','bond','dispersion')

    def selection(heavy, nao):
        chosen = [l for l in train if all(sizes[i]['heavy']<=heavy and
                  (nao is None or sizes[i]['nao_tz']<=nao) for i in l['terms'])]
        ids = {i for l in chosen for i in l['terms']}
        molecular = {i for i in ids if sizes[i]['natom']>1}
        curves = Counter(l['curve'] for l in chosen if l['category']=='dispersion')
        if any(n != 10 for n in curves.values()):
            raise ValueError('Size selection broke a D442 distance curve')
        by_category = {}
        for category in categories:
            subset = [l for l in chosen if l['category']==category]
            selected = {i for l in subset for i in l['terms'] if sizes[i]['natom']>1}
            by_category[category] = dict(labels=len(subset), molecular_states=len(selected),
                closed_shell=sum(states[i]['spin']==0 for i in selected))
        closed = {i for i in molecular if states[i]['spin']==0}
        result = dict(max_heavy=heavy, max_nao_tz=nao, labels=len(chosen),
                      molecular_states=len(molecular), closed_shell=len(closed),
                      open_shell=len(molecular-closed), atomic_states=len(ids-molecular),
                      dispersion_curves=len(curves), by_category=by_category,
                      shared_with_heldout=len(molecular & heldout),
                      closed_shell_shared_with_heldout=len(closed & heldout),
                      closed_shell_train_exclusive=len(closed-heldout),
                      open_shell_train_exclusive=len(molecular-closed-heldout),
                      elements=sorted({e for i in molecular for e in states[i]['symbols']}),
                      largest_nao_tz=max((sizes[i]['nao_tz'] for i in molecular),default=0),
                      largest_nao_dz=max((sizes[i]['nao_dz'] for i in molecular),default=0))
        return result, chosen, ids

    rows = [selection(h,None)[0] for h in (2,3,4)]
    rows += [selection(3,n)[0] for n in (150,200,250)]
    selected, chosen, ids = selection(max_heavy,max_nao)
    def make_cases(observations):
        uses = defaultdict(list)
        for label in observations:
            for i,coefficient in label['terms'].items():
                uses[i].append(dict(label=label['id'], category=label['category'],
                                    coefficient=coefficient, curve=label.get('curve')))
        return [dict(name='ixc2_'+i, state_id=i,
                     atom=list(zip(states[i]['symbols'],states[i]['geometry_angstrom'])),
                     unit='Angstrom', charge=states[i]['charge'], spin=states[i]['spin'],
                     source='IXC2 train energy-observation constituents',
                     shared_with_heldout=i in heldout, size=sizes[i], uses=uses[i])
                for i in sorted(uses)]

    cases = make_cases(chosen)
    groups = dict(
        closed_shell=[c for c in cases if c['size']['natom']>1 and c['spin']==0 and not c['shared_with_heldout']],
        open_shell=[c for c in cases if c['size']['natom']>1 and c['spin']!=0 and not c['shared_with_heldout']],
        shared_with_heldout=[c for c in cases if c['size']['natom']>1 and c['shared_with_heldout']],
        atomic_references=[c for c in cases if c['size']['natom']==1])
    # A dimer combines two monomer bases. A separate 300-AO budget admits more
    # chemically diverse small complexes without admitting the many larger TAE25 molecules.
    nci_labels = [l for l in train if l['category']=='dispersion' and
                  all(sizes[i]['nao_tz']<=300 for i in l['terms'])]
    expanded_labels = [l for l in chosen if l['category']!='dispersion']+nci_labels
    expanded_cases = make_cases(expanded_labels)
    groups['closed_shell_dispersion300'] = [c for c in expanded_cases
        if c['size']['natom']>1 and c['spin']==0 and not c['shared_with_heldout']]
    expanded_curves = Counter(l['curve'] for l in nci_labels)
    if any(n!=10 for n in expanded_curves.values()):
        raise ValueError('Expanded selection broke a D442 curve')
    report = dict(format='dcc-ixc2-screen-v1', run_root=str(run_root), database=str(db_path),
                  database_sha256=checksum, train_population=population,
                  database_labels=len(labels), database_states=len(states),
                  train_referenced_states=len({i for l in train for i in l['terms']}),
                  screening_code_sha256=digest(__file__), basis_counts=basis_counts,
                  size_sweep=rows, selected=selected,
                  exported_cases={name:len(data) for name,data in groups.items()},
                  expanded_dispersion=dict(max_nao_tz=300, max_heavy=None,
                      curves=len(expanded_curves), energy_observations=len(nci_labels),
                      closed_shell=len(groups['closed_shell_dispersion300']),
                      largest_nao_tz=max(c['size']['nao_tz'] for c in groups['closed_shell_dispersion300']),
                      largest_nao_dz=max(c['size']['nao_dz'] for c in groups['closed_shell_dispersion300']),
                      note='Alternative complete closed-shell list, not additional to closed_shell.json; TAE25/BSE49 limits unchanged.'),
                  density_labels_computed=False,
                  identity_note='Counts use existing state IDs (geometry, charge, spin); no rotation/permutation deduplication.',
                  split_note='Selection uses train observations only. Constituents shared with held-out observations are exported separately.',
                  coverage_notes=[
                      'BSE49 supplies parent/radical endpoint geometries, not stretched-bond paths.',
                      'D442x10 is dispersion-focused; it does not establish broad hydrogen-bond coverage.',
                      'Size screening is not an H3C accuracy or convergence certification.'])
    # Retain the raw D442 names/tags so the small-complex coverage can be audited.
    coverage = []
    nci = db_path.parent/'raw/NCIA_D442x10.zip'
    if nci.exists():
        with zipfile.ZipFile(nci) as z:
            names = dict(line.split('\t',1) for line in z.read('NCIA_D442x10/NCIA_D442x10_system_names.txt').decode().splitlines() if '\t' in line)
            tags = {row[0]:row[1:] for line in z.read('NCIA_D442x10/NCIA_D442x10_metadata.txt').decode().splitlines()
                    if len(row:=line.split('\t'))==3 and row[0]!='system'}
        for curve in sorted({l['curve'] for l in chosen if l['category']=='dispersion'}):
            key=curve+'_100'
            coverage.append(dict(curve=curve, name=names[key], group=tags[key][0], tags=tags[key][1]))
        report['expanded_dispersion']['complexes']=[
            dict(curve=curve,name=names[curve+'_100'],group=tags[curve+'_100'][0])
            for curve in sorted(expanded_curves)]
        report['nci_archive_sha256']=digest(nci)
    report['selected_dispersion_complexes']=coverage
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+f'.{os.getpid()}.tmp')
    members={'report.json':json.dumps(report,indent=2).encode(),
             'selected-energy-observations.jsonl':('\n'.join(json.dumps(l,separators=(',',':')) for l in chosen)+'\n').encode()}
    members['expanded-dispersion-energy-observations.jsonl']=(
        '\n'.join(json.dumps(l,separators=(',',':')) for l in expanded_labels)+'\n').encode()
    for name, data in groups.items():
        members['cases/'+name+'.json']=json.dumps(data,separators=(',',':')).encode()
    # A compact standalone record of the exact input and selection algorithm.
    members['provenance/run.json']=(run_root/'run.json').read_bytes()
    members['provenance/dataset.py']=Path(__file__).read_bytes()
    for p in sorted((db_path.parent/'licenses').glob('*')):
        if p.is_file(): members['licenses/'+p.name]=p.read_bytes()
    try:
        with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
            for name,raw in members.items():z.writestr(name,raw)
            z.writestr('index.json',json.dumps({n:hashlib.sha256(b).hexdigest() for n,b in members.items()},indent=2))
        with zipfile.ZipFile(tmp) as z:
            for name,raw in members.items():
                if z.read(name)!=raw:raise ValueError('Screening archive verification failed')
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return report


# Pinned public geometries. These are energy benchmarks, not density labels.
NCIA_REVISION = '1816bfc72609d7deb1d4f93ab9e27eb13bb44bec'
PSI4_REVISION = 'c3cd35420200a5c7122930b517c339056e763c85'
GMTKN55_REVISION = '8d485b37a1ca8837e395042671ca5ba4e0714691'
GMTKN55_FILES = {
    'IL16/008/struc.xyz': '67c5ee89d4760aa7cdd2caf48af1627f7bac6cbbdeb013c7d032a34edf4b8041',
    'IL16/144/struc.xyz': 'e597e4068d65d753c540d8a4062eda8f4d066aa4963e10e41290759968d6cc5c',
    'IL16/147/struc.xyz': 'bd3358942f5dff41d382068e8ce1ed2a8a44e5d7e5bc0bb0a630d004c13cd219',
    'IL16/148/struc.xyz': '30f6463d578faca9b9eb0db3a5fa763968d438b8ad81b145e0592b56f44be250',
    'IL16/150/struc.xyz': '21f5e49cfb5af065d3e74af6448fba3027205cfec8fefb05915fecea052c6e0f',
    'IL16/152/struc.xyz': 'bb27d33d6223e639d4a4d718c4f1ede67b9048be3b8c89632b83ece7105ee107',
    'IL16/187/struc.xyz': '6eda61f67add90672868a2c663240b1ac4e9a748471cc647da4b01b14df9dfd6',
    'IL16/202/struc.xyz': '3d170e8cb16455bedec4da19b6ff394184dcd4dbda3d7db84844b45723b8802b',
    'IL16/212/struc.xyz': 'f74545147bb95e868ec18b5c387646e92e7123729d5a5fc104893f6ff0fe94ea',
    'IL16/213/struc.xyz': 'e38b94d2adcc20abd8372345fc2bfc353a2d7d7648e5d079d88ab08e6a7b6910',
    'IL16/214/struc.xyz': '6c5b081d9c29127b2eb9dd19f8ac780e2199eea33328c1ab003849ff7d2fb588',
    'IL16/227/struc.xyz': '905f6a2934ea157443a634fc0e3169f9cedce656bf4ff4e4a92fcf70c242ebb0',
    'IL16/228/struc.xyz': 'e04b7a23930dd67dc73ba44adf8f925034781b9856b29560eed0aef5849cef9f',
    'IL16/229/struc.xyz': '27ed522fbd623eb53b6729acd3a1c92a23b22db43f5f3b069d1e2a13beef2612',
    'IL16/230/struc.xyz': 'd2f2e4db19cde020b7da33a211e1f81ddd3fe6c443609e9dd13a6e8515bc0a0d',
    'IL16/231/struc.xyz': 'fc402556e090465802060d4ebcc9ffc43ad5390254522620df6cfb1d3594483d',
    'IL16/227A/struc.xyz': '085a9e6027184f9c4f12fcefc9ec51335f02ac77805dfb8cdbe0ee4717798f9b',
    'IL16/227B/struc.xyz': '2c855a3c458b2644de92b6d53ddff0fb2a7f1959fc99557a070af3c9069f916b',
    'IL16/227/.CHRG': '9a271f2a916b0b6ee6cecb2426f0b3206ef074578be55d9bc94f6f3fe3ab86aa',
    'IL16/227A/.CHRG': '4355a46b19d348dc2f57c046f8ef63d4538ebb936000f3c9ee954a27460dd865',
    'IL16/227B/.CHRG': 'ee3aa64bb94a50845d5024cd4bd20202a4567aed5cd5328c0d97e9920775fc28',
    'LICENSE': '29853614116e554998fb67ca0654c69dadedaf70960344027acdfa8ee42f0a68',
    'IL16/.bib': '9d53713473bd6355c1b7fab7b807267a704ffb491ee255b76115c61e74a62cc0',
    'IL16/.res': 'aaad5fddf0ac7475b6caed1196d7faf8dda597fd0b323fcded6dcfe88b44b063',
}
NCI_SOURCES = {
    'HB375x10': ('3ec52600dbf83b48729795858615a0c0e9411a5c21354c3e453c966f35f285af', '10.1021/acs.jctc.9b01265'),
    'IHB100x10': ('4e15f1456eb9fba28dd7e2cca4cb4cfdd19712863ce9a513fc8b0e92bea93fdd', '10.1021/acs.jctc.9b01265'),
    'HB300SPXx10': ('cfa420253be63395ef3548de9b389af15bc3a03feb185a8a22e942fef085d120', '10.1021/acs.jctc.0c00715'),
    'SH250x10': ('88fcf5feef0e431f4600ccba27fca7bd3d0e5738d9fe595a68229de8556fe88e', '10.1039/D2CP01600A'),
    'R739x5': ('0f6f122dcebe61e508ee1dccd4890ceb4329449d827fe361196de90eae0e1176', '10.1021/acs.jctc.0c01341'),
}


def _small_sources(directory, previous, base):
    """Read pinned bytes locally, from a previous output, or from public URLs."""
    nci_root = f'https://raw.githubusercontent.com/Honza-R/NCIAtlas/{NCIA_REVISION}'
    psi_root = f'https://raw.githubusercontent.com/psi4/psi4/{PSI4_REVISION}'
    specs = {name+'.zip': (f'{nci_root}/packaged_sets/NCIA_{name}_github_package.zip', sha)
             for name, (sha, _) in NCI_SOURCES.items()}
    specs.update({
        'A24.py': (f'{psi_root}/psi4/share/psi4/databases/A24.py',
                   '1cac58beca09b348e5f129e80accfea2ea6dd28a14f8cebedb41ee5ce292f757'),
        'README.md': (f'{nci_root}/README.md',
                      '76da4c4bc6732607781c911312f31d977d96efa6c5257d91db48d1f6fda3d2d5'),
        'Psi4-COPYING': (f'{psi_root}/COPYING',
                         '8ceb4b9ee5adedde47b31e975c1d90c73ad27b6b165a1dcd80c7c545eb65b903'),
        'Psi4-COPYING.LESSER': (f'{psi_root}/COPYING.LESSER',
                                'da7eabb7bafdf7d3ae5e9f223aa5bdc1eece45ac569dc21b3b037520b4464768'),
    })
    specs.update({'GMTKN55-'+path.replace('/', '_'):
                  (f'https://raw.githubusercontent.com/grimme-lab/GMTKN55/{GMTKN55_REVISION}/{path}', sha)
                  for path, sha in GMTKN55_FILES.items()})
    embedded = {}
    archive = previous if previous.exists() else INPUT
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            embedded = {Path(n).name: z.read(n) for n in z.namelist() if n.startswith('sources/')}
    contents, provenance = {}, {}
    for filename, (url, sha) in specs.items():
        if directory is not None:
            raw = (Path(directory)/filename).read_bytes()
        elif filename in embedded:
            raw = embedded[filename]
        else:
            with urllib.request.urlopen(url, timeout=90) as response:
                raw = response.read()
        if hashlib.sha256(raw).hexdigest() != sha:
            raise ValueError(f'Source checksum mismatch: {filename}')
        contents[filename] = raw
        provenance[filename] = dict(url=url, sha256=sha, bytes=len(raw))
    raw = Path(base).read_bytes() if Path(base).exists() else embedded['ixc2-small.zip']
    sha = '41c12d3c7c3fcffa76ac6d4c0d61add6ba99961e71fb1eaa5189dba1de9f1892'
    if hashlib.sha256(raw).hexdigest() != sha:
        raise ValueError('IXC2 base differs from the reviewed small-molecule selection')
    contents['ixc2-small.zip'] = raw
    provenance['ixc2-small.zip'] = dict(sha256=sha, bytes=len(raw),
        source='Reviewed IXC2 train selection; original provenance and licenses inside this ZIP')
    return contents, provenance


def _nci_curves(raw, name):
    """Parse source XYZ and fragment charges without executing downloaded code."""
    curves = defaultdict(list)
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        prefix = f'NCIA_{name}/NCIA_{name}'
        names = dict(line.split('\t', 1) for line in
                     z.read(prefix+'_system_names.txt').decode().splitlines() if '\t' in line)
        tags = {r[0]: r[1:] for line in z.read(prefix+'_metadata.txt').decode().splitlines()
                if len(r := line.split('\t')) == 3 and not r[0].startswith('#') and r[0] != 'system'}
        for member in sorted(z.namelist()):
            if not member.endswith('.xyz'):
                continue
            lines = z.read(member).decode().splitlines()
            fields = dict(item.split('=', 1) for item in lines[1].split())
            atoms = [[r[0], list(map(float, r[1:4]))] for line in lines[2:] if (r := line.split())]
            if len(atoms) != int(lines[0]):
                raise ValueError(f'Truncated geometry: {member}')
            key = Path(member).stem
            group, tagged = tags[key]
            fragments = []
            for letter in ('a', 'b'):
                bounds = list(map(int, fields['selection_'+letter].split('-')))
                start, end = bounds[0], bounds[-1]
                fragments.append(dict(indices=list(range(start-1, end)),
                                      charge=int(fields['charge_'+letter]), spin=0))
            if sorted(i for f in fragments for i in f['indices']) != list(range(len(atoms))):
                raise ValueError(f'Invalid fragment partition: {member}')
            if sum(f['charge'] for f in fragments) != int(fields['charge']):
                raise ValueError(f'Invalid fragment charges: {member}')
            curve = key.rsplit('_', 1)[0]
            curves[curve].append(dict(name=f'ncia_{name}_{key}', atom=atoms, unit='Angstrom',
                charge=int(fields['charge']), spin=0, source=name, source_member=member,
                description=names[key], family=f'{name}/{curve}', role='dimer', group=group,
                tags=tagged.strip('"').split(','), scaling=float(fields['scaling']), fragments=fragments,
                source_energy=dict(value=float(fields['benchmark_Eint']), unit=fields['benchmark_unit'],
                                   kind='published interaction energy, not a density label')))
    for curve, points in curves.items():
        expected = [1., 1.05, 1.10, 1.15, 1.25] if name == 'R739x5' else [.8, .85, .9, .95, 1., 1.05, 1.1, 1.25, 1.5, 2.]
        if sorted(p['scaling'] for p in points) != expected:
            raise ValueError(f'Incomplete source curve: {name}/{curve}')
    return curves


def _a24_cases(raw):
    """Read literal geometry blocks with AST; do not import the Psi4 module."""
    geometries, descriptions, energies = {}, {}, {}
    for node in ast.parse(raw.decode()).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
            continue
        key = target.slice
        if not isinstance(key, ast.BinOp) or not isinstance(key.op, ast.Mod):
            continue
        if not isinstance(key.right, ast.Tuple) or len(key.right.elts) != 2:
            continue
        ident = int(ast.literal_eval(key.right.elts[1]))
        fmt = ast.literal_eval(key.left)
        if target.value.id == 'GEOS' and fmt == '%s-%s-dimer':
            if not isinstance(node.value, ast.Call) or len(node.value.args) != 1:
                raise ValueError('Unexpected A24 geometry syntax')
            geometries[ident] = ast.literal_eval(node.value.args[0])
        elif target.value.id == 'TAGL' and fmt == '%s-%s':
            descriptions[ident] = ast.literal_eval(node.value).strip()
        elif target.value.id == 'BIND':
            energies[ident] = ast.literal_eval(node.value)
    if set(geometries) != set(range(1, 25)):
        raise ValueError('Incomplete A24 geometry source')
    cases = []
    for ident, text in sorted(geometries.items()):
        if 'units angstrom' not in text.lower():
            raise ValueError('Unexpected A24 coordinate unit')
        atoms, fragments = [], []
        for fragment in text.lower().replace('units angstrom', '').split('--'):
            rows = [line.split() for line in fragment.splitlines() if line.strip()]
            charge, mult = map(int, rows[0])
            start = len(atoms)
            atoms.extend([[r[0].capitalize(), list(map(float, r[1:]))] for r in rows[1:]])
            fragments.append(dict(indices=list(range(start, len(atoms))), charge=charge, spin=mult-1))
        if len(fragments) != 2 or any(f['spin'] for f in fragments):
            raise ValueError('Unexpected open-shell A24 fragment')
        group = ('hydrogen_bond' if ident <= 5 else 'mixed_polar' if ident <= 13 else
                 'dispersion' if ident <= 21 else 'repulsive_pi')
        cases.append(dict(name=f'a24_{ident:02d}', atom=atoms, unit='Angstrom',
            charge=sum(f['charge'] for f in fragments), spin=0, source='A24',
            description=descriptions[ident], family=f'A24/{ident:02d}', role='dimer',
            group=group, scaling=1., fragments=fragments, source_member=f'A24.py:GEOS[A24-{ident}-dimer]',
            source_energy=dict(value=energies[ident], unit='kcal/mol',
                               kind='published interaction energy, not a density label')))
    return cases


def prepare_small(output=INPUT, sources=None, base=None):
    """Build an offline, provenance-complete small-molecule input archive.

    Preserve published dimer coordinates and complete distance curves. Select
    monomers at each curve's 1.0 point, without ghosts or reoptimization.
    No quantum chemistry is run and no density accuracy is inferred.
    """
    output = Path(output)
    base = Path(base) if base else Path(__file__).resolve().parent/'data/ixc2-small.zip'
    raw_sources, provenance = _small_sources(sources, output, base)
    elements = 'H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar'.split()
    counts = {basis: {e: sum((2*b[0]+1)*(len(b[1])-1) for b in gto.basis.load(basis, e))
                      for e in elements} for basis in ('aug-cc-pvdz', 'aug-cc-pvtz')}

    def size(case):
        symbols = [a[0] for a in case['atom']]
        xyz = np.asarray([a[1] for a in case['atom']], dtype=float)
        if not symbols or not set(symbols) <= set(elements) or xyz.shape != (len(symbols), 3) or not np.isfinite(xyz).all():
            raise ValueError(f'Invalid atom data: {case["name"]}')
        if case['unit'] != 'Angstrom' or case['spin'] != 0:
            raise ValueError('Production pool must be singlet and in Angstrom')
        nelec = sum(gto.charge(e) for e in symbols)-case['charge']
        if nelec <= 0 or nelec % 2:
            raise ValueError(f'Invalid singlet electron count: {case["name"]}')
        if len(symbols) > 1:
            distances = np.linalg.norm(xyz[:, None]-xyz[None, :], axis=2)
            if distances[np.triu_indices(len(symbols), 1)].min() < .2:
                raise ValueError(f'Overlapping atoms: {case["name"]}')
        return dict(natom=len(symbols), heavy=sum(e != 'H' for e in symbols),
                    nao_dz=sum(counts['aug-cc-pvdz'][e] for e in symbols),
                    nao_tz=sum(counts['aug-cc-pvtz'][e] for e in symbols))

    with zipfile.ZipFile(io.BytesIO(raw_sources['ixc2-small.zip'])) as z:
        cases = json.loads(z.read('cases/closed_shell_dispersion300.json'))
        open_cases = json.loads(z.read('cases/open_shell.json'))
        atoms = json.loads(z.read('cases/atomic_references.json'))
        base_report = json.loads(z.read('report.json'))
    for case in cases:
        if size(case) != case['size'] or case['shared_with_heldout']:
            raise ValueError('Invalid reviewed IXC2 case')
    base_count = len(cases)
    audit, selected_curves, dimers, monomers = {}, [], [], []
    for name in NCI_SOURCES:
        curves = _nci_curves(raw_sources[name+'.zip'], name)
        candidates = {}
        for key, points in curves.items():
            if any(not set(a[0] for a in p['atom']) <= set(elements) for p in points):
                continue
            for p in points:
                p['size'] = size(p)
            if len({p['size']['nao_tz'] for p in points}) != 1:
                raise ValueError('Source curve changes composition')
            candidates[key] = next(p for p in points if p['scaling'] == 1.)
        def category(p):
            return p['group']+'/'+p['tags'][0] if name in ('SH250x10', 'R739x5') else p['group']
        buckets = defaultdict(list)
        for key, point in candidates.items():
            buckets[category(point)].append(key)
        selected = {k for k, p in candidates.items() if p['size']['nao_tz'] <= 300}
        if name == 'R739x5':
            # The smallest available system for each element-contact class.
            selected = {min((k for k in keys if k in selected),
                            key=lambda k: (candidates[k]['size']['nao_tz'], k))
                        for keys in buckets.values() if any(k in selected for k in keys)}
        else:
            # Fill missing classes with one smallest representative, never >400 AO.
            for keys in buckets.values():
                if not any(k in selected for k in keys):
                    best = min(keys, key=lambda k: (candidates[k]['size']['nao_tz'], k))
                    if candidates[best]['size']['nao_tz'] <= 400:
                        selected.add(best)
        coverage = Counter(category(candidates[k]) for k in selected)
        audit[name] = dict(source_curves=len(curves), first_three_period_curves=len(candidates),
            selected_curves=len(selected), selected_dimers=sum(len(curves[k]) for k in selected),
            coverage=dict(sorted(coverage.items())),
            unavailable_classes=sorted(set(buckets)-set(coverage)),
            exceptions_above_300=[candidates[k]['name'] for k in sorted(selected)
                                  if candidates[k]['size']['nao_tz'] > 300])
        for key in sorted(selected):
            reference = candidates[key]
            points = sorted(curves[key], key=lambda p: p['scaling'])
            # Report internal-distance changes rather than silently merging
            # approximately equal fragment geometries from different points.
            max_change = 0.
            for fragment in reference['fragments']:
                idx = fragment['indices']
                coords = np.array([reference['atom'][i][1] for i in idx])
                ref_dist = np.linalg.norm(coords[:, None]-coords[None, :], axis=2)
                for p in points:
                    xyz = np.array([p['atom'][i][1] for i in idx])
                    max_change = max(max_change, float(np.max(np.abs(
                        np.linalg.norm(xyz[:, None]-xyz[None, :], axis=2)-ref_dist))))
            selected_curves.append(dict(family=reference['family'], description=reference['description'],
                category=category(reference), nao_tz=reference['size']['nao_tz'],
                dimer_names=[p['name'] for p in points],
                monomer_internal_distance_max_change_angstrom=max_change))
            dimers.extend(points)
    a24 = _a24_cases(raw_sources['A24.py'])
    for case in a24:
        case['size'] = size(case)
        if case['size']['nao_tz'] > 400:
            raise ValueError('A24 case exceeds size budget')
    dimers.extend(a24)
    # IL16 adds an actual cation-anion pair, distinct from ion-neutral IHB.
    def il_xyz(key):
        lines = raw_sources[f'GMTKN55-IL16_{key}_struc.xyz'].decode().splitlines()
        atoms = [[r[0], list(map(float, r[1:]))] for line in lines[2:] if (r := line.split())]
        if len(atoms) != int(lines[0]):
            raise ValueError('Truncated IL16 XYZ')
        return atoms
    il_sizes = {path.split('/')[1]: sum(counts['aug-cc-pvtz'][a[0]] for a in il_xyz(path.split('/')[1]))
                for path in GMTKN55_FILES if re.fullmatch(r'IL16/\d+/struc.xyz', path)}
    if sorted(k for k, n in il_sizes.items() if n <= 400) != ['227']:
        raise ValueError('IL16 size selection changed')
    ion_pair, cation, anion = il_xyz('227'), il_xyz('227A'), il_xyz('227B')
    charges = [int(raw_sources[f'GMTKN55-IL16_{key}_.CHRG']) for key in ('227', '227A', '227B')]
    if ion_pair[:-1] != cation or ion_pair[-1][0] != anion[0][0] or charges != [0, 1, -1]:
        raise ValueError('Unexpected IL16 fragment mapping or charges')
    energy_line = next(line for line in raw_sources['GMTKN55-IL16_.res'].decode().splitlines()
                       if line.startswith('$tmer 227{'))
    ion_case = dict(name='gmtkn55_il16_227', atom=ion_pair, unit='Angstrom', charge=charges[0], spin=0,
        source='GMTKN55-IL16', source_member='GMTKN55-IL16_227_struc.xyz',
        description='ethylammonium ... chloride', family='IL16/227', role='dimer',
        group='cation_anion_pair', scaling=1., fragments=[
            dict(indices=list(range(len(cation))), charge=charges[1], spin=0),
            dict(indices=[len(cation)], charge=charges[2], spin=0)],
        source_energy=dict(value=float(energy_line.split()[-1]), unit='kcal/mol',
                           kind='published interaction energy, not a density label'))
    ion_case['size'] = size(ion_case)
    dimers.append(ion_case)
    for case in dimers:
        if case['scaling'] != 1.:
            continue
        for label, fragment in zip(('a', 'b'), case['fragments']):
            mono = dict(name=case['name']+'_mono_'+label,
                atom=[case['atom'][i] for i in fragment['indices']], unit='Angstrom',
                charge=fragment['charge'], spin=fragment['spin'], source=case['source'],
                source_member=case['source_member'], family=case['family'], role='monomer',
                parent=case['name'], fragment=label, group=case['group'])
            mono['size'] = size(mono)
            (monomers if len(mono['atom']) > 1 else atoms).append(mono)
    cases.extend(dimers+monomers)
    # Exact Cartesian/charge/spin identity only. No approximate, rotational or
    # permutation merging: all published input geometries remain recoverable.
    unique, identities, aliases = [], {}, {}
    for case in cases:
        key = json.dumps([case['atom'], case['charge'], case['spin']], separators=(',', ':'))
        if key in identities:
            aliases[case['name']] = identities[key]
        else:
            identities[key] = case['name']
            unique.append(case)
    cases = unique
    if len({c['name'] for c in cases}) != len(cases):
        raise ValueError('Duplicate production case name')

    # Explicitly derived covalent scans are a separate research challenge.
    challenge = []
    for symbols in (('H','H'), ('Li','H'), ('H','F'), ('H','Cl'), ('N','N'), ('C','O'), ('F','F'), ('Cl','Cl')):
        anchors = [c for c in cases if sorted(a[0] for a in c['atom']) == sorted(symbols)
                   and c['charge'] == 0 and c['spin'] == 0]
        if not anchors:
            raise ValueError(f'Missing covalent scan anchor: {symbols}')
        anchor = min(anchors, key=lambda c: (not any(u['category']=='atomization' for u in c.get('uses', [])), c['name']))
        for scale in (.8, .9, 1., 1.1, 1.25, 1.5, 1.75, 2., 2.5, 3.):
            xyz = np.asarray([a[1] for a in anchor['atom']])
            stretched = xyz[0]+scale*(xyz-xyz[0])
            challenge.append(dict(name='stretch_'+'_'.join(symbols)+'_'+str(scale).replace('.', 'p'),
                atom=[[a[0], r.tolist()] for a, r in zip(anchor['atom'], stretched)], unit='Angstrom',
                charge=0, spin=0, size=anchor['size'], source='Derived rigid diatomic bond scaling',
                parent=anchor['name'], family='stretch/'+'-'.join(symbols), scaling=scale,
                recommended_for_production=False,
                note='Singlet branch only; dissociation can be multireference. H3C accuracy unverified.'))
    for case in challenge:
        size(case)
    report = dict(format='dcc-small-selection-v1', density_labels_computed=False,
        production_count=len(cases), base_ixc2_count=base_count, added_source_dimers=len(dimers),
        added_source_monomers=len(monomers), exact_duplicates_removed=len(aliases),
        atomic_reference_records=len(atoms), open_shell_candidates=len(open_cases),
        derived_bond_challenge=len(challenge), sources=provenance, nci_selection=audit,
        a24=dict(dimers=len(a24), groups=dict(Counter(c['group'] for c in a24))),
        il16=dict(source_dimers=16, selected=['227'], nao_tz_by_system=il_sizes,
                  rule='One ion-pair representative within the <=400 AO exception budget'),
        charge_counts=dict(Counter(str(c['charge']) for c in cases)),
        source_counts=dict(Counter(c['source'] for c in cases)),
        nao_tz=dict(max=max(c['size']['nao_tz'] for c in cases),
            quantiles=dict(zip(('p50','p90','p95','p99'),
                np.quantile([c['size']['nao_tz'] for c in cases], [.5,.9,.95,.99]).tolist())),
            above_300=sum(c['size']['nao_tz']>300 for c in cases)),
        max_nao_dz=max(c['size']['nao_dz'] for c in cases), basis_counts=counts,
        selection_rules=[
            'IXC2 TAE25/BSE49: <=3 heavy atoms and aTZ<=200; D442 complete curves with aTZ<=300.',
            'NCIA: H-Ar, aTZ<=300; one cheapest <=400 exception per otherwise missing group/contact.',
            'R739: one cheapest <=300 curve per (group, contact); preserve all five points.',
            'A24: all 24 published geometries; <=400 AO, no generated NCI coordinates.',
            'IL16: smallest cation-anion complex (ethylammonium chloride), aTZ=372; others >400.',
            'New monomers: extract only at scaling=1.0, without ghosts or relaxation.',
            'Deduplication: exact Cartesian geometry, charge and spin; aliases retained.'],
        split_note='Production candidate pool, not an independent accuracy test. New sources have no train/test assignment. Keep related curves/monomers together when splitting; source overlap is possible.',
        limitations=['No density accuracy or cost certification from geometry screening.',
            'No aromatic stacking, transition metals, solvent ensembles or general reaction/TS coverage.',
            'Bond-stretch challenge and open-shell inputs are excluded from default production shards.'],
        base_selection=base_report['selected'], builder_sha256=digest(__file__))
    members = {'sources/'+n: raw for n, raw in raw_sources.items()}
    def add_json(name, value):
        members[name] = (json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)+'\n').encode()
    add_json('cases/production.json', cases)
    add_json('cases/open_shell.json', open_cases)
    add_json('cases/bond_stretch_challenge.json', challenge)
    add_json('cases/atomic_references.json', atoms)
    add_json('selection/curves.json', selected_curves)
    add_json('selection/exact_aliases.json', aliases)
    add_json('report.json', report)
    members['ATTRIBUTION.txt'] = (
        'DCC small-molecule input selection. No density labels are included.\n'
        'NCI Atlas: J. Rezac, K. Kriz and coauthors; CC BY 4.0.\n'
        'https://creativecommons.org/licenses/by/4.0/\n'
        'See sources/README.md for authors and papers; selection/curves.json records selected systems.\n'
        'A24: J. Rezac and P. Hobza, https://doi.org/10.1021/ct400057w\n'
        'Geometry source: unmodified Psi4 A24.py, LGPL-3.0, with COPYING and COPYING.LESSER.\n'
        'GMTKN55: Goerigk et al., https://doi.org/10.1039/C7CP04913G, CC BY 4.0.\n'
        'IL16: Lao, Schaffer, Jansen and Herbert, https://doi.org/10.1021/ct5010593; original files retained.\n'
        'IXC2 constituents: sources/ixc2-small.zip contains the TAE25, BSE49 and D442 provenance/licenses.\n'
        'Changes: size selection, format conversion, fragment extraction, exact duplicate aliases.\n'
        'Derived diatomic scans are separately marked and excluded from production.\n'
    ).encode()
    # Round-robin AO-size distribution is a scheduling heuristic, not a runtime prediction.
    shards = [[] for _ in range(min(64, len(cases)))]
    for i, case in enumerate(sorted(cases, key=lambda c: (-c['size']['nao_tz'], c['name']))):
        shards[i % len(shards)].append(case)
    specs = []
    for i, chunk in enumerate(shards):
        chunk.sort(key=lambda c: (c['size']['nao_tz'], c['name']))
        member = f'dcc/inputs/{i:04d}.jsonl'
        raw = ('\n'.join(json.dumps(c, sort_keys=True, separators=(',', ':')) for c in chunk)+'\n').encode()
        members[member] = raw
        specs.append(dict(member=member, count=len(chunk), sha256=hashlib.sha256(raw).hexdigest()))
    add_json('dcc/metadata.json', dict(format='dcc-small-input-v1', count=len(cases), shards=specs,
        density_labels_included=False, selection_report='report.json',
        scheduling='64 shards, AO-size round robin; one molecule per process/node, multiple molecules per job'))
    add_json('index.json', {n: hashlib.sha256(raw).hexdigest() for n, raw in members.items()})
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_name(output.name+f'.{os.getpid()}.tmp')
    try:
        with zipfile.ZipFile(tmp, 'w') as z:
            for name, raw in sorted(members.items()):
                info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                z.writestr(info, raw, compresslevel=6)
        with zipfile.ZipFile(tmp) as z:
            for name, raw in members.items():
                if z.read(name) != raw:
                    raise ValueError('Prepared archive verification failed')
        prepare_input(tmp)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return report
