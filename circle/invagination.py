"""Hollow epithelial-ring prototype: cortical tensions and a compliant lumen.

Primary deformation is followed only until the first proposed topology event.
Polarized tension inheritance across rearrangements is not implemented here;
no event is silently suppressed and evolution is not continued past that point.
"""
import numpy as np
from activefoam.foam_full import boundary_normal_signs
from .disc import DiscFoam,WallRecord
from .tiling import DiscTiling
from .kernels import advance_disc


class TopologyLimit(RuntimeError):pass
class LumenRupture(ValueError):pass


def annular_tiling(n=24,inner_radius=3.,spacing=.075,wall_factor=1.3):
    outer_radius=np.sqrt(inner_radius**2+n/np.pi)
    wall=wall_factor*outer_radius;center=np.full(2,4*wall)
    angle=2*np.pi*np.arange(n)/n
    radial=np.column_stack((np.cos(angle),np.sin(angle)))
    pos=np.vstack((center+inner_radius*radial,center+outer_radius*radial))
    cells=[];arcs={}
    for i in range(n):
        j=(i+1)%n;cells.append([i,n+i,n+j,j])
        for radius,offset in [(inner_radius,0),(outer_radius,n)]:
            count=max(2,int(np.ceil(radius*2*np.pi/n/spacing))+1)
            a=np.linspace(angle[i],angle[i]+2*np.pi/n,count)
            arcs[i+offset,j+offset]=center+radius*np.column_stack((np.cos(a),np.sin(a)))
    return DiscTiling(pos,cells,arcs,wall,center,0,2*n),outer_radius


def boundary_loops(f):
    """Oriented free boundaries: tissue on the left, independent of radius."""
    free=set(map(int,np.flatnonzero(f.e_f[:,0]<0)));outgoing={};oriented={}
    for c,loop in enumerate(f.f_e):
        for signed in loop:
            e=signed if signed>=0 else -signed-1
            if e not in free:continue
            if f.e_f[e,1]!=c:raise LumenRupture('ambiguous free boundary')
            a,b=map(int,f.e_v[e] if signed>=0 else f.e_v[e][::-1])
            if a in outgoing:raise LumenRupture('branched free boundary')
            outgoing[a]=(b,e);oriented[e]=np.array(f.e_mid[e] if signed>=0 else f.e_mid[e][::-1])
    remaining=set(free);loops=[]
    while remaining:
        e=min(remaining);start=next(v for v,(_,edge) in outgoing.items() if edge==e)
        vertex=start;edges=[];points=[]
        for _ in range(len(free)+1):
            if vertex not in outgoing:raise LumenRupture('open boundary')
            other,e=outgoing[vertex]
            if e not in remaining:raise LumenRupture('boundary repeated before closure')
            remaining.remove(e);edges.append(e);points.extend(oriented[e][:-1]);vertex=other
            if vertex==start:break
        else:raise LumenRupture('boundary did not close')
        points=np.array(points);local=points-f.center
        area=.5*np.sum(local[:,0]*np.roll(local[:,1],-1)-local[:,1]*np.roll(local[:,0],-1))
        loops.append(dict(edges=edges,points=points,area=float(area)))
    outer=[x for x in loops if x['area']>0];holes=[x for x in loops if x['area']<0]
    if len(outer)!=1 or len(holes)!=1:raise LumenRupture('prototype requires one intact shell and one closed lumen')
    return outer[0],holes[0]


class PolarFoam(DiscFoam):
    STATE_PARAMETERS=('lumen_reference_area','lumen_initial_pressure','lumen_bulk_modulus','active_cells',
                      'apical_amplitude','lateral_amplitude','activation_start','lateral_delay')
    STATE_PHYSICAL=('morph_time',)

    @classmethod
    def ring(cls,n=24,inner_radius=3.,spacing=.075,dt=.0025,seed=1,bulk=5.):
        tiling,outer=annular_tiling(n,inner_radius,spacing)
        f=cls(tiling,w=1.,rho=1.,edpc=spacing,dt=dt,seed=seed)
        _,inside=boundary_loops(f)
        f.lumen_reference_area=-inside['area']
        # A stated initial pressure scale, not a measured embryonic parameter.
        f.lumen_initial_pressure=1/outer+1/inner_radius;f.lumen_bulk_modulus=bulk
        theta=2*np.pi*(np.arange(n)+.5)/n
        delta=np.angle(np.exp(1j*(theta+np.pi/2)))
        f.active_cells=np.abs(delta)<np.pi/6
        f.apical_amplitude=0.;f.lateral_amplitude=0.;f.activation_start=0.;f.lateral_delay=0.;f.morph_time=0.
        return f

    def lumen_pressure(self,area):
        if area<=0:raise LumenRupture('nonpositive lumen area')
        return self.lumen_initial_pressure+self.lumen_bulk_modulus*(self.lumen_reference_area/area-1)

    def interface_data(self):
        outer,inner=boundary_loops(self);area=-inner['area'];p_l=self.lumen_pressure(area)
        cell_pressure=self.P0*(self.A0/self.f_area-1)/self.lsc
        edge_pressure=np.zeros(len(self.e_f));count=np.sum(self.e_f>=0,axis=1)
        fixed=count-self.gam*np.all(self.e_f>=0,axis=1)
        for e,(a,b) in enumerate(self.e_f):
            edge_pressure[e]=(cell_pressure[b] if b>=0 else 0)-(cell_pressure[a] if a>=0 else 0)
        edge_pressure[inner['edges']]-=p_l
        if self.morph_time>=self.activation_start:
            for e in outer['edges']:
                if self.active_cells[self.e_f[e,1]]:fixed[e]+=self.apical_amplitude
        if self.morph_time>=self.activation_start+self.lateral_delay:
            for e,(a,b) in enumerate(self.e_f):
                if a>=0 and b>=0:fixed[e]+=self.lateral_amplitude*.5*(int(self.active_cells[a])+int(self.active_cells[b]))
        return outer,inner,area,p_l,edge_pressure,fixed,count

    def step(self,mu=0.):
        _,_,_,_,edge_pressure,fixed,count=self.interface_data()
        mids,loops=self._fast_geometry_inputs();outward=boundary_normal_signs(self.e_f,loops)
        (vertices,new,counts,excursion,residual,resampled,v_react,v_direct,v_flag,v_trib,
         n_edge,n_index,n_pos,n_react,n_direct,n_flag,n_trib)=advance_disc(
            self.vpos,self.e_v,self.e_f,self.e_t,mids,self.f_area,self.psi,self.dt,
            self.P0,self.lsc,self.edpc,outward,self.center[0],self.center[1],self.radius,
            self.wall_tol,self.remesh,self.wall_resample,edge_pressure)
        if excursion>self.delta_max:raise ValueError('trial point exceeds auxiliary-box guarantee')
        self.vpos,self.e_mid,self.e_rfn=vertices,new,counts
        self.wall=WallRecord(vertices.copy(),v_react,v_direct,v_flag,v_trib,n_edge,n_index,n_pos,
            n_react,n_direct,n_flag,n_trib,float(excursion),float(residual),int(resampled),self.center.copy(),self.radius)
        xi=self.rng.standard_normal(len(count))
        self.e_t=self.e_t-self.dt/self.tauT*(self.e_t-fixed)+mu/self.tauT*np.sqrt(self.dt)/2*count*xi
        self._wrap_vertices();self._update_faces();self.morph_time+=self.dt

    def _guard(self,kind,ident,site,operation,accepted_of,rejected_value):
        raise TopologyLimit(f'first proposed {kind}; polarized transition inheritance requires validation')

    def frame(self):
        outer,inner,area,pressure,_,_,_=self.interface_data()
        return dict(time=self.morph_time,n=self.nFa,polygons=[self._cell_polyline(c) for c in range(self.nFa)],
                    outer=outer['points'],inner=inner['points'],lumen_area=area,lumen_pressure=pressure,
                    center=self.center.copy(),wall_radius=self.radius,active_cells=self.active_cells.copy(),
                    apical_amplitude=self.apical_amplitude,lateral_amplitude=self.lateral_amplitude)
