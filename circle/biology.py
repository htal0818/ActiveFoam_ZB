"""Explicit mechanosensitive tension and growth extensions of the disk model.

No change to the original force law: per-cell preferred areas generalize K/A;
feedback adds a specified term to the tension drift. Global length/mobility units
remain those of the founding cells, even after division.
"""
import copy
import numpy as np
from .disc import DiscFoam
from . import state


class BiologyFoam(DiscFoam):
    STATE_PARAMETERS=('feedback_gain','feedback_delay','feedback_scale','memory_spacing','growth_rate')
    STATE_PHYSICAL=('preferred','cell_uid','parent_uid','birth_time','next_uid','bio_time',
                    'feedback_pairs','feedback_reference','feedback_history','feedback_times',
                    'feedback_cursor','feedback_next_sample')
    STATE_DIAGNOSTICS=('division_log',)

    @classmethod
    def from_disc(cls,foam,gain=0.,delay=0.,strain_scale=.1,memory_spacing=.05,growth_rate=0.):
        if type(foam) is not DiscFoam:raise TypeError('start from a circular DiscFoam')
        if min(gain,delay,growth_rate)<0 or strain_scale<=0 or memory_spacing<=0:raise ValueError('invalid biology parameters')
        if not np.isclose(memory_spacing/foam.dt,round(memory_spacing/foam.dt)):raise ValueError('memory spacing must be a multiple of dt')
        new=object.__new__(cls)
        for key,value in vars(foam).items():setattr(new,key,state._copy(value))
        new.rng=copy.deepcopy(foam.rng);new.log=copy.deepcopy(foam.log)
        new.feedback_gain=float(gain);new.feedback_delay=float(delay);new.feedback_scale=float(strain_scale)
        new.memory_spacing=float(memory_spacing);new.growth_rate=float(growth_rate)
        new.preferred=np.full(new.nFa,new.A0)
        new.cell_uid=np.arange(new.nFa,dtype=np.int64);new.parent_uid=np.full(new.nFa,-1,dtype=np.int64)
        new.birth_time=np.zeros(new.nFa);new.next_uid=new.nFa;new.bio_time=0.
        new.feedback_pairs=[];new.feedback_reference=np.empty(0)
        rows=int(np.ceil(delay/memory_spacing))+3
        new.feedback_history=np.empty((rows,0));new.feedback_times=np.arange(-rows,0,dtype=float)*memory_spacing
        new.feedback_cursor=0;new.feedback_next_sample=0.;new.division_log=[]
        new._feedback_signal()
        return new

    @property
    def nominal_density(self):return float(self.preferred.sum()/self.disc_area)

    def contact_lengths(self):
        """Total length of all films between one pair of persistent cell IDs."""
        result={}
        for faces,length in zip(self.e_f,self.edge_lengths()):
            if np.all(faces>=0):
                key=tuple(sorted(int(self.cell_uid[f]) for f in faces))
                result[key]=result.get(key,0.)+length
        return result

    def _feedback_signal(self):
        """Bounded delayed strain; zero prehistory for newly formed contacts."""
        lengths=self.contact_lengths();pairs=sorted(lengths)
        if pairs!=self.feedback_pairs:
            previous={key:i for i,key in enumerate(self.feedback_pairs)}
            history=np.zeros((len(self.feedback_times),len(pairs)));reference=np.empty(len(pairs))
            for i,key in enumerate(pairs):
                if key in previous:
                    old=previous[key];reference[i]=self.feedback_reference[old];history[:,i]=self.feedback_history[:,old]
                else:reference[i]=max(lengths[key],1e-12)
            self.feedback_pairs=pairs;self.feedback_reference=reference;self.feedback_history=history
        strain=np.array([lengths[key] for key in pairs])/self.feedback_reference-1 if pairs else np.empty(0)
        if self.bio_time+1e-10>=self.feedback_next_sample:
            i=self.feedback_cursor;self.feedback_times[i]=self.bio_time;self.feedback_history[i]=strain
            self.feedback_cursor=(i+1)%len(self.feedback_times)
            self.feedback_next_sample+=self.memory_spacing
        if self.feedback_delay==0:delayed=strain
        else:
            order=np.argsort(self.feedback_times);times=self.feedback_times[order];hist=self.feedback_history[order]
            # Append the current value if it lies beyond the last stored sample.
            if self.bio_time>times[-1]+1e-12:
                times=np.append(times,self.bio_time);hist=np.vstack((hist,strain))
            target=self.bio_time-self.feedback_delay
            if target<=times[0]:delayed=hist[0]
            else:
                i=min(np.searchsorted(times,target,side='right')-1,len(times)-2)
                f=(target-times[i])/(times[i+1]-times[i]);delayed=(1-f)*hist[i]+f*hist[i+1]
        values=np.tanh(delayed/self.feedback_scale)
        return dict(zip(pairs,values))

    def step(self,mu=0.):
        signal=self._feedback_signal() if self.feedback_gain else {}
        if self.growth_rate:self.preferred*=np.exp(self.growth_rate*self.dt)
        physical_area=self.f_area
        # Passing A*A0/A0_i to the unchanged kernel yields P0*(A0_i/A-1)/lsc.
        # The uniform baseline bypasses this transformation for exact parity.
        if not np.all(self.preferred==self.A0):self.f_area=physical_area*self.A0/self.preferred
        try:super().step(mu)
        except BaseException:
            self.f_area=physical_area
            raise
        if self.feedback_gain:
            extra=np.zeros(len(self.e_f))
            for e,faces in enumerate(self.e_f):
                if np.all(faces>=0):extra[e]=signal[tuple(sorted(int(self.cell_uid[f]) for f in faces))]
            self.e_t+=self.dt/self.tauT*self.feedback_gain*extra
        self.bio_time+=self.dt

    def normal_force(self,ei,eln,areas=None):
        areas=self.f_area if areas is None else areas
        pn=0.
        for side,c in enumerate(self.e_f[ei]):
            if c>=0:pn+=(1 if side else -1)*self.P0*(self.preferred[c]/areas[c]-1)/self.lsc
        return pn*eln/2

    def divide_synchronously(self):
        from .division import divide_round
        return divide_round(self)
