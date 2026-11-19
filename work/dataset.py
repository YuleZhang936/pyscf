"""QM9/IXC2 input, lossless DCC-TZ density storage, and restartable ZIP shards.

Each closed-shell label stores two signed symmetric float64 AO matrices:
the aTZ MP2 base and the aDZ (CCSD - MP2) correction. The matrices keep their
own basis definitions; real-space density is evaluated when requested.
"""
import bz2
from collections import Counter, defaultdict
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import zipfile

import numpy as np
from pyscf import gto
from pyscf.dcc_tz import density_diagnostics, evaluate_density

INPUT = Path(__file__).resolve().parent/'data/qm9.zip'
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
    """Keep the original dataset and add 128 independently readable case lists."""
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
