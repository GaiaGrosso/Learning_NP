import numpy as np
from scipy.stats import chi2

import NNrun

class Point1:
    """
    object describing a point candidate for stage 1 (interval selection)
    """
    def __init__(self, wc, architecture, NtrialsMax, NtrialsMin, t_calculator, logscale=True, printer=False):
        if not type(architecture)==type([]):
            raise Exception('architecture must be a list of layers size')
        if NtrialsMin>NtrialsMax:
            raise Exception('NtrialsMax must be greater than NtrialsMin')
        if NtrialsMin<=0:
            raise Exception('NtrialsMin must be stricly greater than zero')
        self.t_list       = []
        self.t_mean       = 0
        self.t_error      = 0
        self.delta        = 0
        self.Ntrials      = 0
        self.NtrialsMin   = int(NtrialsMin)
        self.NtrialsMax   = int(NtrialsMax)
        self.wc           = wc
        self.logwc        = np.log(wc)
        self.architecture = architecture
        ndoftarget        = 0
        for i in range(len(architecture)-1):
            ndoftarget += (architecture[i]+1)*architecture[i+1]
        self.ndoftarget   = ndoftarget
        self.status       = ''
        self.t_calculator = t_calculator # function(architecture, wc)
        self.logscale     = logscale
        self.printer      = printer
        if logscale:
            self.mean_target = np.log(ndoftarget)
        else:
            self.mean_target = ndoftarget
        self.init_point()
        #self.is_uptodate  = True   
        return
    
    def update_t_list(self, t):
        self.t_list.append(t)
        return
    
    
    def update_Ntrials(self):
        self.Ntrials = len(self.t_list)
        return
    
    def update_t_mean(self):
        if self.logscale: 
            self.t_mean = np.mean(np.log(self.t_list))
        else:
            self.t_mean = np.mean(self.t_list)
        return
    
    def update_t_error(self):
        if self.logscale:
            self.t_error = np.sqrt(np.var(np.log(self.t_list))*1./self.Ntrials)
        else:
            self.t_error = np.sqrt(np.var(self.t_list)*1./self.Ntrials)
        return
    
    def update_delta(self):
        self.delta = (self.t_mean-self.mean_target)/self.t_error
        return
    
    def update_status(self):
        if self.delta < -4 or self.delta > 4:
            self.status = 'killit'
        elif self.delta < -3 and self.delta > -4:
            self.status = 'Islowerbound'
        elif self.delta > 3 and self.delta < 4:
            self.status = 'Isupperbound'
        else :
            self.status = 'Istop'
        return
    
    def plot_point(self):
        print('WC candidate:    '+str(self.wc))
        print('Ntrials updated: '+str(self.Ntrials))
        print('t_mean updated:  '+str(self.t_mean))
        print('t_error updated: '+str(self.t_error))
        print('delta updated:   '+str(self.delta))
        print('status updated:  '+str(self.status))
        return
    
    def init_point(self):
        #for i in range(self.NtrialsMin):
        while self.Ntrials < self.NtrialsMin:
            t = self.t_calculator(self.architecture, self.wc)
            self.update_t_list(t)
            self.update_Ntrials()
            
        self.update_t_mean()
        self.update_t_error()
        self.update_delta()
        self.update_status()
        if self.printer:
            self.plot_point()
        return
    
    def add_trial(self):
        if self.Ntrials < self.NtrialsMin:
            self.init_point()
        elif self.Ntrials >= self.NtrialsMax:
            if self.printer:
                print('WC candidate: '+str(self.wc))
                print('Point status: '+self.status)
                print('Ntrials has reached its maximum.')
            return
        else:
            t = self.t_calculator(self.architecture, self.wc)
            self.update_t_list(t)
            self.update_Ntrials()
            self.update_t_mean()
            self.update_t_error()
            self.update_delta()
            self.update_status()
            if self.printer:
                self.plot_point()
        return
    
    
#####################################################################################################    


class InitialPointsFinder:
    """
    Class of methods to find the optimal WC range for a given architecture.
    Is is based on the 
    
    """
    def __init__(self, wc0, architecture, NtrialsMax, NtrialsMin, t_calculator, wc_update_method='secant', logscale=True):
        if not type(architecture) == type([]):
            raise Exception('architecture must be a list of layers size')
        
        self.Points        = []
        self.wc            = wc0
        self.architecture  = architecture
        ndoftarget         = 0
        for i in range(len(architecture)-1):
            ndoftarget += (architecture[i]+1)*architecture[i+1]
        self.ndoftarget    = ndoftarget
        self.NtrialsMin    = int(NtrialsMin)
        self.NtrialsMax    = int(NtrialsMax)
        self.t_calculator  = t_calculator
        self.logscale      = logscale
        self.Istop_found   = False
        self.Isupper_found = False
        self.Islower_found = False
        self.Plowerbound   = 0
        self.Pupperbound   = 0
        self.Ptop          = 0
        self.bisec_flag    = True
        self.Pointc_brent  = 0
        self.Pointd_brent  = 0
        self.tolerance     = 0
        if wc_update_method == 'secant':
            self.update_wc_method = self.update_wc_secant
        if wc_update_method == 'brent':
            self.update_wc_method = self.update_wc_brent
        return
    
    def update_wc_secant(self, x0, x1, f0, f1, targetPoint='center'):
        wc_new = np.exp((x0*f1-x1*f0)/(f1-f0))
        return wc_new
    
    def update_wc_brent(self, x0, x1, f0, f1, targetPoint='center'):
        if self.tolerance == 0:
            self.tolerance = self.Points[-1].t_error
            
        if f0*f1 > 0:
            wc_new = self.update_wc_secant(x0, x1, f0, f1)
        else:
            # b is the best candidate between the two
            if np.abs(f0)<np.abs(f1):
                b  = x0
                a  = x1
                fb = f0
                fa = f1
                if self.Pointc_brent == 0:
                    print('Start using brent algorithm')
                    self.Pointc_brent = self.Points[-1]
                    c  = a
                    fc = fa
                    d  = 0
                    fd = 0
                else:
                    if targetPoint   == 'center':
                        c  = self.Pointc_brent.logwc
                        fc = self.Pointc_brent.t_mean - self.Pointc_brent.mean_target
                        d  = self.Pointd_brent.logwc
                        fd = self.Pointd_brent.t_mean - self.Pointd_brent.mean_target
                    elif targetPoint == 'upperbound':
                        c  = self.Pointc_brent.logwc
                        fc = self.Pointc_brent.t_mean - (self.Pointc_brent.mean_target + 3*self.Pointc_brent.t_error)
                        d  = self.Pointd_brent.logwc
                        fd = self.Pointd_brent.t_mean - (self.Pointd_brent.mean_target + 3*self.Pointd_brent.t_error)
                    elif targetPoint == 'lowerbound':
                        c  = self.Pointc_brent.logwc
                        fc = self.Pointc_brent.t_mean - (self.Pointc_brent.mean_target - 3*self.Pointc_brent.t_error)
                        d  = self.Pointd_brent.logwc
                        fd = self.Pointd_brent.t_mean - (self.Pointd_brent.mean_target - 3*self.Pointd_brent.t_error)
                    
            else:
                b  = x1
                a  = x0
                fb = f1
                fa = f0
                if self.Pointc_brent == 0:
                    print('Start using brent algorithm')
                    self.Pointc_brent = self.Points[-2]
                    c  = a
                    fc = fa
                    d  = 0
                    fd = 0
                else:
                    if targetPoint   == 'center':
                        c  = self.Pointc_brent.logwc
                        fc = self.Pointc_brent.t_mean - self.Pointc_brent.mean_target
                        d  = self.Pointd_brent.logwc
                        fd = self.Pointd_brent.t_mean - self.Pointd_brent.mean_target
                    elif targetPoint == 'upperbound':
                        c  = self.Pointc_brent.logwc
                        fc = self.Pointc_brent.t_mean - (self.Pointc_brent.mean_target + 3*self.Pointc_brent.t_error)
                        d  = self.Pointd_brent.logwc
                        fd = self.Pointd_brent.t_mean - (self.Pointd_brent.mean_target + 3*self.Pointd_brent.t_error)
                    elif targetPoint == 'lowerbound':
                        c  = self.Pointc_brent.logwc
                        fc = self.Pointc_brent.t_mean - (self.Pointc_brent.mean_target - 3*self.Pointc_brent.t_error)
                        d  = self.Pointd_brent.logwc
                        fd = self.Pointd_brent.t_mean - (self.Pointd_brent.mean_target - 3*self.Pointd_brent.t_error)
            
            # secant method
            s = b -fb*(b-a)/(fb-fa)
            # bisection method
            condition1 = not (s>b and s<(3*a+b)/4.)
            condition2 = self.bisec_flag and (np.abs(s-b)>=0.5*np.abs(b-c))
            condition3 = (not self.bisec_flag) and (np.abs(b-c))
            condition4 = self.bisec_flag and (np.abs(b-c)<self.tolerance)
            condition5 = (not self.bisec_flag) and (np.abs(c-d)<self.tolerance) and (not self.Pointd_brent == 0)
            if condition1 and condition2 and condition3 and condition4 and condition5:
                s = (a+b)/2.
                self.bisec_flag = True
            else: 
                self.bisec_flag = False
           
            self.Pointd_brent = self.Pointc_brent
            if np.abs(f0)<np.abs(f1):
                self.Pointc_brent = self.Points[-2]
            else:
                self.Pointc_brent = self.Points[-1]
            wc_new = np.exp(s)        
            print(self.Pointc_brent.t_mean, self.Pointd_brent.t_mean)
        return wc_new
    
    def update_wc(self, method, targetPoint='center'):
        if len(self.Points)==0:
            # wc0 has never been tested
            wc_new = self.wc
        elif len(self.Points)==1:
            # rescale wc towards the right direction
            p      = self.Points[-1]
            wc_new = np.exp(p.logwc*p.mean_target/p.t_mean)
           
        else:
            # secant method from the last two points in Points[]
            x0     = self.Points[-2].logwc
            x1     = self.Points[-1].logwc
            if targetPoint == 'center':
                f0  = self.Points[-2].t_mean - self.Points[-2].mean_target
                f1  = self.Points[-1].t_mean - self.Points[-1].mean_target
            elif targetPoint == 'upperbound':
                f0  = self.Points[-2].t_mean - (self.Points[-2].mean_target + 3*self.Points[-2].t_error)
                f1  = self.Points[-1].t_mean - (self.Points[-1].mean_target + 3*self.Points[-1].t_error)
            elif targetPoint == 'lowerbound':
                f0  = self.Points[-2].t_mean - (self.Points[-2].mean_target - 3*self.Points[-2].t_error)
                f1  = self.Points[-1].t_mean - (self.Points[-1].mean_target - 3*self.Points[-1].t_error)
            wc_new = method(x0, x1, f0, f1, targetPoint='center')
            
        self.wc = wc_new
        return
    
    def PointCenterFinder(self):
        while not self.Istop_found:
            self.update_wc(self.update_wc_method, 'center')
            p = Point1(self.wc, self.architecture, self.NtrialsMax, self.NtrialsMin, self.t_calculator)
            while p.Ntrials < p.NtrialsMax:
                print('Point: '+str(len(self.Points))+', Ntrials: '+str(p.Ntrials))
                if p.status == 'Istop' and p.Ntrials == p.NtrialsMax-1:
                    self.Istop_found = True
                    break
                else:
                    p.add_trial()
                    if p.status == 'Islowerbound' and p.Ntrials == p.NtrialsMax-1:
                        self.Islower_found == True
                        self.Plowerbound = p
                    elif p.status == 'Isupperbound' and p.Ntrials == p.NtrialsMax-1:
                        self.Isupper_found == True
                        self.Pupperbound = p
            self.Points.append(p)
            p.plot_point()
            self.Ptop = p
        return
    
    def PointUpperFinder(self):
        if self.Isupper_found == True:
            print('Upper bound has already been found.')
            return
        else:
            while not self.Isupper_found:
                self.update_wc(self.update_wc_method, 'upperbound')
                p = Point1(self.wc, self.architecture, self.NtrialsMax, self.NtrialsMin, self.t_calculator)
                while p.Ntrials < p.NtrialsMax:
                    if p.status == 'Isupperbound' and p.Ntrials == p.NtrialsMax-1:
                        self.Isupper_found = True
                        break
                    else:
                        p.add_trial()
                        if p.status == 'Islowerbound' and p.Ntrials == p.NtrialsMax-1:    
                            self.Islower_found == True
                            self.Plowerbound = p
            
            self.Points.append(p)
            p.plot_point()
            self.Pupperbound = p
            return
        
    def PointLowerFinder(self):
        if self.Islower_found == True:
            print('Lower bound has already been found.')
            return
        else:
            while not self.Islower_found:
                self.update_wc(self.update_wc_method, 'lowerbound')
                p = Point1(self.wc, self.architecture, self.NtrialsMax, self.NtrialsMin, self.t_calculator)
                while p.Ntrials < p.NtrialsMax:
                    if p.status == 'Islowerbound' and p.Ntrials == p.NtrialsMax-1:
                        self.Islower_found = True
                        break
                    else:
                        p.add_trial()
            
            self.Points.append(p)
            p.plot_point()
            self.Plowerbound = p
            return
        
    def plot_points(self):
        self.Ptop.plot_point()
        self.Pupperbound.plot_point()
        self.Plowerbound.plot_point()
        print('---------------------\n')
        return
    
