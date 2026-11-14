"""Instantaneous, atomic synchronous cleavage of every current cell.

Existing films are split without moving nodes. A new straight daughter contact is
inserted inside each mother. No integration or random draws occur between splits.
"""
import numpy as np
from activefoam.foam_full import edge_mid_vrtx_average
from . import state,protocol


class DivisionRejected(ValueError):pass


def area(p):return .5*np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))


def clip_polygon(p,normal,offset):
    points=[]
    for a,b in zip(p,np.roll(p,-1,axis=0)):
        da=float(a@normal-offset);db=float(b@normal-offset)
        if da<=0:points.append(a)
        if (da<0<db) or (db<0<da):points.append(a+da/(da-db)*(b-a))
    return np.array(points)


def bisector(p,normal):
    lo,hi=float((p@normal).min()),float((p@normal).max());target=area(p)/2
    for _ in range(60):
        mid=(lo+hi)/2;part=clip_polygon(p,normal,mid)
        a=area(part) if len(part)>=3 else 0.
        if a<target:lo=mid
        else:hi=mid
    return (lo+hi)/2


def major_axis(p,uid):
    cross=p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1);a=area(p)
    q=np.roll(p,-1,axis=0)
    center=np.array([np.sum((p[:,j]+q[:,j])*cross)/(6*a) for j in (0,1)])
    v=p-center;w=np.roll(v,-1,axis=0);cross=v[:,0]*w[:,1]-v[:,1]*w[:,0]
    xx=np.sum(cross*(v[:,0]**2+v[:,0]*w[:,0]+w[:,0]**2))/(12*a)
    yy=np.sum(cross*(v[:,1]**2+v[:,1]*w[:,1]+w[:,1]**2))/(12*a)
    xy=np.sum(cross*(2*v[:,0]*v[:,1]+v[:,0]*w[:,1]+w[:,0]*v[:,1]+2*w[:,0]*w[:,1]))/(24*a)
    values,vectors=np.linalg.eigh([[xx,xy],[xy,yy]])
    if values[1]-values[0]<1e-6*values[1]:return (uid*2.399963229728653)%(2*np.pi)
    n=vectors[:,1];return float(np.arctan2(n[1],n[0]))


def intersections(foam,cell,normal,offset):
    result=[];tolerance=max(1e-8,.02*foam.edpc)
    for signed in foam.f_e[cell]:
        e=signed if signed>=0 else -signed-1;mid=np.array(foam.e_mid[e]);d=mid@normal-offset
        for k in range(len(mid)-1):
            if (d[k]<=0<d[k+1]) or (d[k+1]<=0<d[k]):
                frac=d[k]/(d[k]-d[k+1]);point=mid[k]+frac*(mid[k+1]-mid[k]);position=k+frac
                if any(np.linalg.norm(point-x[2])<1e-9 for x in result):continue
                if min(np.linalg.norm(point-mid[0]),np.linalg.norm(point-mid[-1]))<tolerance:
                    raise DivisionRejected('cleavage too close to an existing junction')
                result.append((e,position,point))
    if len(result)!=2:raise DivisionRejected('cleavage must intersect the mother boundary exactly twice')
    if np.linalg.norm(result[0][2]-result[1][2])<foam.edpc:raise DivisionRejected('daughter contact too short')
    return result


def plans(foam):
    output=[];reserved=[];separation=max(1e-8,.04*foam.edpc)
    for cell in range(foam.nFa):
        polygon=foam._cell_polyline(cell);theta=major_axis(polygon,int(foam.cell_uid[cell]))
        accepted=None
        for turn in (0.,.07,-.07,.17,-.17,.31,-.31,.53,-.53,.83,-.83,1.2):
            n=np.array([np.cos(theta+turn),np.sin(theta+turn)]);offset=bisector(polygon,n)
            try:cuts=intersections(foam,cell,n,offset)
            except DivisionRejected:continue
            if any(np.linalg.norm(point-previous)<separation for _,_,point in cuts for previous in reserved):continue
            accepted=(n,offset,theta+turn,turn);reserved.extend(cut[2] for cut in cuts);break
        if accepted is None:raise DivisionRejected(f'no valid cleavage for mother UID {foam.cell_uid[cell]}')
        output.append(accepted)
    return output


def split_film(foam,e,cuts):
    """Split an old film at ordered (fractional-index, point) positions."""
    old=np.array(foam.e_mid[e]);endpoints=foam.e_v[e].copy();faces=foam.e_f[e].copy();tension=foam.e_t[e]
    cuts=sorted(cuts,key=lambda x:x[0]);vertices=[foam._add_vertex(point) for _,point in cuts]
    bounds=[(0.,old[0])]+cuts+[(float(len(old)-1),old[-1])]
    ends=[int(endpoints[0])]+vertices+[int(endpoints[1])];edge_ids=[]
    for i,((start,p),(stop,q)) in enumerate(zip(bounds[:-1],bounds[1:])):
        interior=[old[k] for k in range(len(old)) if start+1e-10<k<stop-1e-10]
        mid=np.vstack([p,*interior,q]);mid=mid[np.r_[True,np.linalg.norm(np.diff(mid,axis=0),axis=1)>1e-12]]
        if len(mid)<2:raise DivisionRejected('empty split film')
        if i==0:
            foam.e_v[e]=[ends[i],ends[i+1]];foam.e_mid[e]=mid;foam.e_rfn[e]=len(mid)-2;edge_ids.append(e)
        else:edge_ids.append(foam._add_edge(ends[i],ends[i+1],*faces,mid,len(mid)-2,tension))
    for c,loop in enumerate(foam.f_e):
        new=[]
        for signed in loop:
            if signed==e:new.extend(edge_ids)
            elif signed==-e-1:new.extend(-k-1 for k in edge_ids[::-1])
            else:new.append(signed)
        foam.f_e[c]=new
    return vertices


def rebuild(foam):
    foam.nFa=len(foam.f_e);nv=len(foam.vpos);ne=len(foam.e_v)
    incident=[[] for _ in range(ne)];foam.f_v=[]
    for c,loop in enumerate(foam.f_e):
        starts=[];stops=[]
        for signed in loop:
            e=signed if signed>=0 else -signed-1;ends=foam.e_v[e] if signed>=0 else foam.e_v[e][::-1]
            starts.append(int(ends[0]));stops.append(int(ends[1]));incident[e].append(c)
        if stops!=starts[1:]+starts[:1]:raise DivisionRejected('daughter boundary is not a closed walk')
        foam.f_v.append(starts)
    faces=[]
    for cells in incident:
        if len(cells)==1:faces.append([-1,cells[0]])
        elif len(cells)==2 and cells[0]!=cells[1]:faces.append(sorted(cells))
        else:raise DivisionRejected('invalid film incidence')
    foam.e_f=np.array(faces,dtype=np.int64)
    edges=[set() for _ in range(nv)];cells=[set() for _ in range(nv)]
    for e,ends in enumerate(foam.e_v):
        for v in ends:edges[v].add(e);cells[v].update(c for c in foam.e_f[e] if c>=0)
    foam.v_e=np.full((nv,3),-1,dtype=np.int64);foam.v_f=foam.v_e.copy()
    for v in range(nv):
        if len(edges[v])>3 or len(cells[v])>3 or not edges[v]:raise DivisionRejected('nontrivalent division junction')
        foam.v_e[v,:len(edges[v])]=sorted(edges[v]);foam.v_f[v,:len(cells[v])]=sorted(cells[v])
    for key in state.KEYED_CACHES+state.STATE_CACHES:
        if hasattr(foam,key):delattr(foam,key)
    foam._update_faces()


def divide_cell(foam,cell,plan):
    n,offset,theta,turn=plan;cuts=intersections(foam,cell,n,offset)
    old_area=foam.f_area[cell];old_preferred=foam.preferred[cell];parent=int(foam.cell_uid[cell])
    by_edge={}
    for e,f,p in cuts:by_edge.setdefault(e,[]).append((f,p))
    cut_vertices=[]
    for e,points in by_edge.items():cut_vertices.extend(split_film(foam,e,points))
    va,vb=cut_vertices;loop=foam.f_e[cell]
    starts=[int(foam.e_v[e if e>=0 else -e-1,0 if e>=0 else 1]) for e in loop]
    a=starts.index(va);loop=loop[a:]+loop[:a];starts=starts[a:]+starts[:a];b=starts.index(vb)
    first,second=loop[:b],loop[b:];daughter=foam.nFa
    mid,count=edge_mid_vrtx_average(foam.vpos[[va,vb]],foam.edpc)
    edge=foam._add_edge(va,vb,cell,daughter,mid,count,2-foam.gam)
    foam.f_e[cell]=first+[-edge-1];foam.f_e.append(second+[edge]);rebuild(foam)
    daughter_area=foam.f_area[cell]+foam.f_area[daughter]
    if abs(daughter_area-old_area)>1e-8*old_area:raise DivisionRejected('division did not conserve physical area')
    fraction=foam.f_area[cell]/daughter_area
    if not .499999<=fraction<=.500001:raise DivisionRejected('division was not equal-area')
    foam.preferred[cell]=old_preferred*fraction;foam.preferred=np.append(foam.preferred,old_preferred*(1-fraction))
    ids=[foam.next_uid,foam.next_uid+1];foam.next_uid+=2
    foam.cell_uid[cell]=ids[0];foam.cell_uid=np.append(foam.cell_uid,ids[1])
    foam.parent_uid[cell]=parent;foam.parent_uid=np.append(foam.parent_uid,parent)
    foam.birth_time[cell]=foam.bio_time;foam.birth_time=np.append(foam.birth_time,foam.bio_time)
    return dict(time=foam.bio_time,parent=parent,daughters=ids,angle=theta,angle_adjustment=turn,
                center=foam.vpos[[va,vb]].mean(axis=0),area_before=old_area,preferred_before=old_preferred)


def divide_round(foam):
    """Double all mothers at one clock time, or roll back the entire round."""
    snap=state.snapshot(foam);count=foam.nFa;total=foam.preferred.sum();physical_area=foam.f_area.sum()
    try:
        prepared=plans(foam)
        events=[divide_cell(foam,c,plan) for c,plan in enumerate(prepared)]
        if foam.nFa!=2*count or not np.isclose(foam.preferred.sum(),total,rtol=1e-13):raise DivisionRejected('division conservation failure')
        if not np.isclose(foam.f_area.sum(),physical_area,rtol=1e-10):raise DivisionRejected('round area conservation failure')
        protocol.validate(foam)
        if foam.rng.bit_generator.state!=snap.rng:raise RuntimeError('division consumed random draws')
        foam._feedback_signal() # newborn cell pairs receive new references and zero prehistory
    except BaseException as exc:
        state.restore(foam,snap)
        foam.division_log.append(dict(time=foam.bio_time,accepted=False,mothers=count,reason=str(exc)))
        if isinstance(exc,ValueError):return False
        raise
    foam.division_log.append(dict(time=foam.bio_time,accepted=True,mothers=count,daughters=foam.nFa,events=events))
    return True
