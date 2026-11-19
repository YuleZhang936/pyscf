"""DCC-TZ (H3C): AE RI-MP2/aTZ + FC (CCSD - RI-MP2)/aDZ.

All three densities are orbital-relaxed energy derivatives. Each signed term
retains its own AO basis. No fitting, basis projection, clipping or renormalizing.
The numerical checks do not certify eta < 1 for an arbitrary molecule.

Usage::

    from pyscf import dcc_tz
    case = dict(name='water', atom='O 0 0 0; H 0 -.757 .587; H 0 .757 .587')
    components = dcc_tz.calculate(case, memory=6000)

The signed components retain their own AO bases; they are not one matrix in
an effective basis. This module has no dependency on dataset or batch code.
"""
from contextlib import contextmanager, nullcontext
import hashlib
import itertools
import math
from pathlib import Path
import time
import types

import numpy as np
import pyscf
from pyscf import cc, dft, gto, lib, scf
from pyscf.cc import ccsd_rdm, uccsd_rdm
from pyscf.grad import ccsd as rg, uccsd as ug
from pyscf.mp.dfmp2_native import DFRMP2
from pyscf.mp.dfump2_native import DFUMP2

ROLES = {'T': ('aug-cc-pvtz', 'ae'), 'A': ('aug-cc-pvdz', 'fc')}
TERMS = [(1, 'T', 'mp2'), (1, 'A', 'ccsd'), (-1, 'A', 'mp2')]
PRODUCTION_SETTINGS = dict(
    profile='pyscf-defaults', scf_tol=scf.hf.SCF.conv_tol,
    scf_grad_tol=scf.hf.SCF.conv_tol_grad,
    scf_direct_tol=scf.hf.SCF.direct_scf_tol,
    max_cycle=scf.hf.SCF.max_cycle,
    cc_tol=cc.ccsd.CCSD.conv_tol, cc_norm_tol=cc.ccsd.CCSD.conv_tol_normt,
    cc_max_cycle=cc.ccsd.CCSD.max_cycle)


def source_version():
    """Fingerprint this PySCF method and its solver dependencies."""
    source = Path(pyscf.__file__).resolve().parent
    files = [Path(__file__).resolve()]
    files += [source/n for n in (
        'scf/hf.py', 'scf/uhf.py', 'scf/addons.py', 'grad/ccsd.py', 'grad/uccsd.py',
        'cc/ccsd.py', 'cc/uccsd.py', 'cc/ccsd_lambda.py', 'cc/uccsd_lambda.py',
        'cc/ccsd_rdm.py', 'cc/uccsd_rdm.py', 'mp/dfmp2_native.py', 'mp/dfump2_native.py')]
    return dict(pyscf_version=pyscf.__version__, numpy_version=np.__version__,
                source_sha256={str(p.relative_to(source.parent)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in files},
                settings=PRODUCTION_SETTINGS, terms=[list(t) for t in TERMS],
                roles={r:list(v) for r,v in ROLES.items()})


def make_mol(case, basis, memory=6000):
    return gto.M(atom=case["atom"], unit=case.get("unit", "Angstrom"),
                 charge=case.get("charge", 0), spin=case.get("spin", 0),
                 basis=basis, symmetry=False, verbose=0, max_memory=memory)



def density_diagnostics(mol, dm):
    if not np.isfinite(dm).all():
        raise ValueError("Non-finite density matrix")
    s = mol.intor_symmetric("int1e_ovlp")
    ne = float(np.einsum("ij,ji", s, dm))
    herm = float(np.max(np.abs(dm - dm.T)))
    if abs(ne - mol.nelectron) > 1e-7 or herm > 1e-9:
        raise ValueError(f"Invalid density: trace={ne}, asymmetry={herm}")
    val, vec = np.linalg.eigh(s)
    shalf = (vec * np.sqrt(np.maximum(val, 0))) @ vec.T
    occ = np.linalg.eigvalsh(shalf @ dm @ shalf)
    # Response densities need not be ensemble N-representable. Record, do not clip.
    return dict(electron_count=ne, trace_error=ne-mol.nelectron,
                hermiticity_error=herm, min_occupation=float(occ.min()),
                max_occupation=float(occ.max()), overlap_min_eigenvalue=float(val.min()))



def core_orbitals(mol, override=None):
    if override is not None:
        count = int(override)
    else:
        count = sum(0 if z<=2 else 1 if z<=10 else 5 for z in mol.atom_charges())
    if not 0 <= count < min(mol.nelec) and count:
        raise ValueError("Default core partition removes an occupied spin space; provide an explicit core partition")
    return count


# Response algebra adapted from PySCF grad/ccsd.py and grad/uccsd.py.
# Copyright 2014-2019 The PySCF Developers. All Rights Reserved.
# Apache License, Version 2.0: https://www.apache.org/licenses/LICENSE-2.0
# Construct the response density without unused nuclear derivative integrals.

def _imat(c, fdm2, keys):
    mol = c.mol
    nao = mol.nao_nr()
    diagonal = np.arange(nao)
    diagonal = diagonal*(diagonal+1)//2+diagonal
    matrices = [np.zeros((nao,nao)) for _ in keys]
    memory = max(0, c.max_memory-lib.current_memory()[0])
    block = max(1, int(memory*.7e6/8/(nao**3*max(2.5,len(keys)+1))))
    p0 = 0
    for b0,b1,nf in rg._shell_prange(mol,0,mol.nbas,block):
        p1 = p0+nf
        eri = mol.intor("int2e", aosym="s2kl", shls_slice=(b0,b1,0,mol.nbas,0,mol.nbas,0,mol.nbas)).reshape(nf,nao,-1)
        for result,key in zip(matrices,keys):
            buf = rg._load_block_tril(fdm2[key],p0,p1,nao)
            buf[:,:,diagonal] *= .5
            result += lib.einsum("ipx,iqx->pq",eri,buf)
        p0 = p1
    return matrices



def relaxed_rdm1(c, eris, l1, l2, unrestricted=False):
    """Return AO spin-summed R density or (alpha,beta) U densities."""
    rdm = uccsd_rdm if unrestricted else ccsd_rdm
    t1,t2 = c.t1,c.t2
    with lib.H5TmpFile() as intermediate, lib.H5TmpFile() as ao_file:
        d1 = rdm._gamma1_intermediates(c,t1,t2,l1,l2)
        d2 = rdm._gamma2_outcore(c,t1,t2,l1,l2,intermediate,True)
        if unrestricted:
            coeff = c.mo_coeff
            indices = [rg._index_frozen_active(mask,occ) for mask,occ in zip(c.get_frozen_mask(),c.mo_occ)]
            active = tuple(mo[:,np.hstack((idx[0],idx[1]))] for mo,idx in zip(coeff,indices))
            ug._rdm2_mo2ao(c,d2,active,ao_file)
            imats = _imat(c,ao_file,["dm2aa+ab","dm2bb+ab"])
            oo,vv = d1[0],d1[3]
            energies, occupations = c._scf.mo_energy,c.mo_occ
            factor = .5
        else:
            coeff = [c.mo_coeff]
            indices = [rg._index_frozen_active(c.get_frozen_mask(),c.mo_occ)]
            idx = indices[0]
            rg._rdm2_mo2ao(c,d2,c.mo_coeff[:,np.hstack((idx[0],idx[1]))],ao_file)
            imats = _imat(c,ao_file,["dm2"])
            oo,vv = [d1[0]],[d1[3]]
            energies, occupations = [c._scf.mo_energy],[c.mo_occ]
            factor = 1.
    overlap = c._scf.get_ovlp()
    dm_mo, imat_mo, nocc = [],[],[]
    for mo, imat, idx, eps, occ, doo, dvv in zip(coeff,imats,indices,energies,occupations,oo,vv):
        oa,va,of,vf = idx
        matrix = -mo.T @ imat @ overlap @ mo
        dm = np.zeros((mo.shape[1],mo.shape[1]))
        dm[np.ix_(oa,oa)] = factor*(doo+doo.T)
        dm[np.ix_(va,va)] = factor*(dvv+dvv.T)
        for frozen,active in ((of,oa),(vf,va)):
            cross = matrix[np.ix_(frozen,active)]/(eps[frozen,None]-eps[active])
            dm[np.ix_(frozen,active)] = cross
            dm[np.ix_(active,frozen)] = cross.T
        dm_mo.append(dm)
        imat_mo.append(matrix)
        nocc.append(np.count_nonzero(occ>0))
    dm_ao = np.array([mo@dm@mo.T for mo,dm in zip(coeff,dm_mo)])
    potentials = c._scf.get_veff(c.mol,dm_ao if unrestricted else dm_ao[0])
    if not unrestricted:
        potentials = [potentials*2]
    rhs = [mo[:,no:].T@v@mo[:,:no] + mat[:no,no:].T-mat[no:,:no]
           for mo,no,v,mat in zip(coeff,nocc,potentials,imat_mo)]
    response = ug._response_dm1(c,tuple(rhs),eris) if unrestricted else [rg._response_dm1(c,rhs[0],eris)]
    correction = np.array([mo@(dm+resp)@mo.T for mo,dm,resp in zip(coeff,dm_mo,response)])
    hf = c._scf.make_rdm1(c.mo_coeff,c.mo_occ)
    return correction+hf if unrestricted else correction[0]+hf


# Keep the validated bounded-memory UCCSD density contraction for open shells.
# QM9 uses the restricted path, which does not install these process-local patches.
MAX_TILE = 2_000_000
LAZY_SIZE = 128_000_000

def dims_for(labels, operands):
    dims = {}
    for labels_i, operand in zip(labels, operands):
        for label, n in zip(labels_i, operand.shape):
            if label in dims and dims[label] != n:
                raise ValueError('Inconsistent Einstein dimensions')
            dims[label] = n
    return dims



def key_slices(slices):
    return tuple((s.start,s.stop,s.step) for s in slices)



class Tensor:
    __array_priority__ = 1000

    def __init__(self, shape, evaluator):
        self.shape = tuple(shape)
        self.ndim = len(shape)
        self.dtype = np.dtype('float64')
        self.evaluator = evaluator

    def block(self, slices, cache=None):
        if cache is None:cache = {}
        key = (id(self), key_slices(slices))
        if key not in cache:cache[key] = self.evaluator(slices,cache)
        return cache[key]

    def __array__(self, dtype=None, copy=None):
        result = np.empty(self.shape, dtype=dtype or self.dtype)
        for i in range(self.shape[0]):
            slices = (slice(i,i+1),) + tuple(slice(0,n) for n in self.shape[1:])
            result[slices] = self.block(slices)
        return result

    def binary(self, other, op):
        if np.isscalar(other):
            return Tensor(self.shape, lambda s,c:op(self.block(s,c),other))
        if tuple(other.shape) != self.shape:
            raise ValueError('Unsupported lazy broadcast')
        if not isinstance(other,Tensor):other=np.array(other,copy=True)
        return Tensor(self.shape, lambda s,c:op(self.block(s,c), fetch(other,s,c)))

    def __add__(self, other):return self.binary(other,np.add)
    def __radd__(self, other):return self+other
    def __sub__(self, other):return self.binary(other,np.subtract)
    def __rsub__(self, other):return -self+other
    def __mul__(self, other):return self.binary(other,np.multiply)
    def __rmul__(self, other):return self*other
    def __truediv__(self, other):return self.binary(other,np.divide)
    def __neg__(self):return self*(-1.)
    def conj(self):return self
    conjugate = conj

    def transpose(self, *axes):
        if len(axes)==1 and isinstance(axes[0],(list,tuple)):axes=tuple(axes[0])
        if not axes:axes=tuple(reversed(range(self.ndim)))
        inverse = np.argsort(axes)
        return Tensor(tuple(self.shape[i] for i in axes),
                      lambda s,c:self.block(tuple(s[i] for i in inverse),c).transpose(axes))

    def reshape(self, *shape):
        if len(shape)==1 and isinstance(shape[0],(list,tuple)):shape=tuple(shape[0])
        if self.ndim==4 and shape==(self.shape[0]*self.shape[1],self.shape[2]*self.shape[3]):
            return PairView(self)
        raise ValueError('Unsupported lazy reshape')

    def __getitem__(self, item):
        if not isinstance(item,tuple):item=(item,)
        item=item+(slice(None),)*(self.ndim-len(item))
        slices=[]; squeeze=[]
        for axis,(index,n) in enumerate(zip(item,self.shape)):
            if isinstance(index,int):slices.append(slice(index,index+1));squeeze.append(axis)
            elif isinstance(index,slice):
                start,stop,step=index.indices(n)
                if step!=1:raise ValueError('Nonunit slice step')
                slices.append(slice(start,stop))
            else:raise ValueError('Unsupported lazy index')
        return np.squeeze(self.block(tuple(slices)),axis=tuple(squeeze))



class PairView:
    def __init__(self, parent):self.parent=parent



def fetch(value, slices, cache):
    return value.block(slices,cache) if isinstance(value,Tensor) else value[tuple(slices)]



def evaluate(spec, operands, output_slices):
    lhs, rhs = spec.split('->')
    labels=lhs.split(','); dims=dims_for(labels,operands)
    selected={k:s for k,s in zip(rhs,output_slices)}
    contracted=sorted(set(''.join(labels))-set(rhs))
    chunks={k:dims[k] for k in contracted}
    lazy=[(lab,operand) for lab,operand in zip(labels,operands) if isinstance(operand,Tensor)]
    def size(lab):
        return math.prod(selected[k].stop-selected[k].start if k in selected else chunks[k] for k in lab)
    while lazy and max(size(lab) for lab,_ in lazy)>MAX_TILE:
        largest=max(lazy,key=lambda item:size(item[0]))[0]
        choices=[k for k in contracted if k in largest and chunks[k]>1]
        if not choices:break
        key=max(choices,key=lambda k:chunks[k]);chunks[key]=(chunks[key]+1)//2
    result=np.zeros(tuple(s.stop-s.start for s in output_slices))
    ranges=[range(0,dims[k],chunks[k]) for k in contracted]
    for starts in itertools.product(*ranges):
        selection=dict(selected,**{k:slice(start,min(dims[k],start+chunks[k])) for k,start in zip(contracted,starts)})
        cache={}
        values=[fetch(op,tuple(selection[k] for k in lab),cache) for lab,op in zip(labels,operands)]
        result+=np.einsum(spec,*values,optimize=True)
    return result



def einsum(spec, *operands, **kwargs):
    if '->' not in spec:raise ValueError('Explicit Einstein output required')
    lhs,rhs=spec.split('->');dims=dims_for(lhs.split(','),operands)
    shape=tuple(dims[k] for k in rhs)
    if len(shape)>=6 or (len(shape)==4 and math.prod(shape)>=LAZY_SIZE):
        # The CCSD RDM algebra subsequently updates several ndarray temporaries
        # in place. Lazy four-index expressions must retain their value at use.
        if len(shape)==4:
            operands=tuple(op if isinstance(op,Tensor) else np.array(op,copy=True) for op in operands)
        return Tensor(shape,lambda s,c:evaluate(spec,operands,s))
    if any(isinstance(op,Tensor) for op in operands):
        return evaluate(spec,operands,tuple(slice(0,n) for n in shape))
    kwargs.setdefault('optimize',True)
    return np.einsum(spec,*operands,**kwargs)



def direct_sum(spec,*operands):
    lhs,rhs=spec.split('->')
    if len(rhs)<6:return lib.direct_sum(spec,*operands)
    labels=lhs.split('+');dims=dims_for(labels,operands)
    shape=tuple(dims[k] for k in rhs)
    def block(slices,cache):
        selected=dict(zip(rhs,slices))
        return lib.direct_sum(spec,*[fetch(op,tuple(selected[k] for k in lab),cache) for op,lab in zip(operands,labels)])
    return Tensor(shape,block)



def take_2d(value, rows, columns, *args, **kwargs):
    if not isinstance(value,PairView):return lib.take_2d(value,rows,columns,*args,**kwargs)
    parent=value.parent
    result=np.empty((len(rows),len(columns)))
    first=np.asarray(rows)//parent.shape[1];second=np.asarray(rows)%parent.shape[1]
    for i in np.unique(first):
        mask=np.where(first==i)[0]
        slab=parent.block((slice(int(i),int(i)+1),)+tuple(slice(0,n) for n in parent.shape[1:]))[0]
        result[mask]=slab[second[mask]].reshape(len(mask),-1)[:,columns]
    return result



class Proxy:
    def __init__(self, original, replacements):self.original,self.replacements=original,replacements
    def __getattr__(self, name):return self.replacements.get(name,getattr(self.original,name))



def clone(function):
    namespace=dict(function.__globals__)
    namespace['numpy']=Proxy(np,{'einsum':einsum})
    namespace['lib']=Proxy(lib,{'direct_sum':direct_sum,'take_2d':take_2d})
    if 'einsum' in namespace:namespace['einsum']=einsum
    return types.FunctionType(function.__code__,namespace,function.__name__,function.__defaults__,function.__closure__)



@contextmanager
def _blocked_uhf():
    original = uccsd_rdm._gamma2_outcore
    uccsd_rdm._gamma2_outcore = clone(original)
    try:
        yield
    finally:
        uccsd_rdm._gamma2_outcore = original


def reference_state(case,basis,memory=10000,unrestricted=None,perturb=None,settings=None):
    settings = PRODUCTION_SETTINGS if settings is None else settings
    mol = make_mol(case,basis,memory)
    unrestricted = bool(case.get("spin",0)) if unrestricted is None else unrestricted
    mf = (scf.UHF if unrestricted else scf.RHF)(mol)
    mf.conv_tol,mf.conv_tol_grad,mf.max_cycle = settings["scf_tol"],settings["scf_grad_tol"],settings["max_cycle"]
    mf.direct_scf_tol = settings["scf_direct_tol"]
    if perturb is not None:
        strength,op = perturb
        h = mf.get_hcore()+strength*op
        mf.get_hcore = lambda *args: h
    mf.kernel()
    if not mf.converged:
        raise RuntimeError("SCF did not converge")
    return mf,unrestricted



def compute(case,basis,methods=("mp2","ccsd"),frozen=0,memory=10000,unrestricted=None,perturb=None,density=True,settings=None):
    if set(methods)-{"mp2","ccsd"}:
        raise ValueError("DCC-TZ only uses MP2 and CCSD")
    settings = dict(PRODUCTION_SETTINGS if settings is None else settings)
    start = time.perf_counter()
    mf,unrestricted = reference_state(case,basis,memory,unrestricted,perturb,settings=settings)
    hf_seconds = time.perf_counter()-start
    mol = mf.mol
    common = dict(case=case,basis=basis,mol=mol.dumps(),frozen=frozen,reference="UHF" if unrestricted else "RHF",
                  nelectron=mol.nelectron,nelec=list(mol.nelec),settings=settings,
                  hf_seconds=hf_seconds,hf_energy=float(mf.e_tot),density_definition="orbital-relaxed energy derivative")
    if density:
        # Retain honest metadata for readers of the older label format.
        common.update(stability_checked=False,internal_stable=None,external_stable=None,
                      spin_square=float(mf.spin_square()[0]) if unrestricted else 0.)
    records = {}
    def record(name,energy,dm,elapsed,extra=None):
        meta = dict(common,method=name,energy=float(energy),post_hf_seconds=elapsed,**(extra or {}))
        spin_dm = dm if unrestricted else np.array([dm*.5,dm*.5]) if dm is not None else None
        total = spin_dm.sum(axis=0) if spin_dm is not None else None
        if total is not None:
            meta.update(density_diagnostics(mol,total))
            s = mol.intor_symmetric("int1e_ovlp")
            spin_trace = np.einsum("sij,ji->s",spin_dm,s)
            if np.max(abs(spin_trace-mol.nelec)) > 1e-7:
                raise RuntimeError("Spin-resolved electron counts are incorrect")
            meta["spin_electron_counts"] = spin_trace.tolist()
        records[name] = dict(meta=meta,dm=total,dm_spin=spin_dm)
    record("hf",mf.e_tot,mf.make_rdm1() if density else None,0.)
    if "mp2" in methods:
        start = time.perf_counter()
        with (DFUMP2 if unrestricted else DFRMP2)(mf,frozen=frozen) as pt:
            pt.kernel()
            dm = pt.make_rdm1(relaxed=True,ao_repr=True) if density else None
            record("mp2",pt.e_tot,dm,time.perf_counter()-start,dict(auxiliary_basis=pt.auxmol.basis,integrals="RI correlation; exact SCF"))
    if "ccsd" in methods:
        start = time.perf_counter()
        c = cc.CCSD(mf,frozen=frozen)
        c.conv_tol,c.conv_tol_normt,c.max_cycle = settings["cc_tol"],settings["cc_norm_tol"],settings.get("cc_max_cycle",settings["max_cycle"])
        eris = c.ao2mo()
        c.kernel(eris=eris)
        if not c.converged:
            raise RuntimeError("CCSD amplitudes did not converge")
        amplitudes_seconds = time.perf_counter()-start
        t1s = c.t1 if unrestricted else [c.t1]
        ncorrelated = sum(c.nocc) if unrestricted else 2*c.nocc
        extra = dict(cc_amplitudes_seconds=amplitudes_seconds,
                     t1_diagnostic=float(np.sqrt(sum(np.linalg.norm(t)**2 for t in t1s)/ncorrelated)),integrals="exact")
        if "ccsd" in methods:
            start = time.perf_counter()
            dm = None
            if density:
                l1,l2 = c.solve_lambda(eris=eris)
                if not c.converged_lambda:
                    raise RuntimeError("CCSD Lambda did not converge")
                dm = relaxed_rdm1(c,eris,l1,l2,unrestricted)
            record("ccsd",c.e_tot,dm,time.perf_counter()-start+amplitudes_seconds,extra)
    return records



def grid(mol, level=5, atom_grid=None):
    grids = dft.gen_grid.Grids(mol)
    grids.level = level
    grids.prune = None
    if atom_grid is not None:
        grids.atom_grid = atom_grid
    grids.build()
    return grids.coords, grids.weights



def evaluate_group(records, coords):
    """Reuse AO evaluations for several methods in the same orbital basis."""
    first = next(iter(records.values()))
    mol = gto.loads(first["meta"]["mol"])
    if any(r["meta"]["mol"] != first["meta"]["mol"] for r in records.values()):
        raise ValueError("Grouped AO evaluation requires identical Mole definitions")
    values = {m: np.empty(len(coords)) for m in records}
    for start in range(0, len(coords), 2048):
        ao = mol.eval_gto("GTOval_sph", coords[start:start+2048])
        for m, record in records.items():
            values[m][start:start+len(ao)] = np.einsum("pi,ij,pj->p", ao, record["dm"], ao, optimize=True)
    return values



def evaluate_density(record, coords, unrelaxed=False):
    mol = gto.loads(record["meta"]["mol"])
    dm = record["unrelaxed"] if unrelaxed else record["dm"]
    if dm is None:
        raise ValueError("Requested density is not present")
    values = np.empty(len(coords))
    for start in range(0, len(coords), 2048):
        ao = mol.eval_gto("GTOval_sph", coords[start:start+2048])
        values[start:start+len(ao)] = np.einsum("pi,ij,pj->p", ao, dm, ao, optimize=True)
    return values



def evaluate_terms(terms,records,coords):
    values={}
    for role in dict.fromkeys(r for _,r,_ in terms):
        methods={m for _,r,m in terms if r==role}
        for method,value in evaluate_group({m:records[role][m] for m in methods},coords).items():
            values[role,method]=value
    return sum(c*values[r,m] for c,r,m in terms)



def calculate(case, memory=6000):
    """Return the three H3C components, sharing the aDZ SCF reference."""
    frozen = core_orbitals(make_mol(case, 'aug-cc-pvtz', memory))
    records = {}
    with _blocked_uhf() if case.get('spin', 0) else nullcontext():
        for role, (basis, core) in ROLES.items():
            methods = sorted({m for _, r, m in TERMS if r == role})
            print(case['name'], role, methods, 'compute', flush=True)
            records[role] = compute(case, basis, methods, frozen=frozen if core == 'fc' else 0,
                                    memory=memory, settings=PRODUCTION_SETTINGS)
    return records


def validate_density(case, records, memory=6000):
    """Check quadrature charge; no HF stability analysis or accuracy claim."""
    mol = make_mol(case, 'aug-cc-pvtz', memory)
    history = []
    for level, atom_grid in [(5,None),(6,None),(7,None),(7,(300,1454)),(7,(500,1454))]:
        coords, weights = grid(mol, level, atom_grid)
        rho = evaluate_terms(TERMS, records, coords)
        error = float(abs(weights @ rho - mol.nelectron))
        history.append(dict(level=level, atom_grid=atom_grid, charge_error=error))
        if np.isfinite(rho).all() and error < 2e-6:
            break
    if not np.isfinite(rho).all() or error >= 2e-6:
        raise RuntimeError('H3C quadrature failed')
    return dict(recipe='H3C', quality_pass=True, grid_history=history,
                source=source_version(), hf_stability_checked=False,
                accuracy_certified=False,
                note='Numerical checks do not establish eta < 1. H3C has an independent C3 failure.')
