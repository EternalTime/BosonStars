import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle
import os

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class bs:
    """
    Finds a spherically symmetric solution to the non linear Klein Gordon
    equation in general relativity for a complex scalar field with a harmonic
    ansatz.

        V(phi) = |phi|^2 + lamb/2 |phi|^4

    The solution is found using a shooting method to find the angular velocity
    eigenvalue of the groudn state. The equations of motion are integrated using
    an RK4 method. For more information see
    gr-qc/0410040
    1202.5809

    Parameters
    ----------
    phi0 :          float
                    Core value of the Boson Star
    lamb :          float, optional
                    Interaction strength, default is 0
    alpha_range :   [float,float], optional
                    Range of the initial values for the lapse function that is
                    used by the shooting method. Default is the full range of
                    [0,1]
    dr :            float, optional
                    Radial step size. Default is .01
    r_max :         float, optional
                    Maximum radius to which the profile will be calculated. The
                    default is 50.
    phi_tol :       float, optional
                    The tolerance of the shooting method. Default is 10^-6
    printTag :      boolean
                    Determines whether to plot the intermediate steps in the
                    shooting routine.

    Returns
    -------
    bs :            Boson Star object
        Methods
        _______
        bs.r :      numpy array
                    radial coordinates
        bs.phi :    numpy array
                    field values at the radial coordinates
        bs.Phi :    numpy array
                    field derivatives at the radial coordinates
        bs.alpha :  numpy array
                    lapse function at the radial coordinates
        bs.a :      numpy array
                    radial metric component at the radial coordinates
        bs.rho :    numpy array
                    energy density at the radial coordinates
        bs.mass :   numpy array
                    mass contained in spheres increasing in radius with the
                    radial coordinate. ADM mass.
        bs.M :      float
                    total mass of the boson star
        bs.R :      float
                    effective radius of the boson star. First moment of the
                    radial coordinate.

    Examples
    --------
    To create a boson star use

        boson_star = bs(phi0)

    This will attempt to create a boson star with a core value of phi0. A
    shooting method is used to pinpoint the correct value of the lapse function
    at the core, and integration is done with an Runge-Kutta 4th order scheme.

    Note that the tolerance is set to 10^-6, so if r_max is too big, the
    profile generator will stop well before the tail of the star is generated.
    In this case lower the maximum radius. Similarly, if r_max is too low, then
    the generator will approach a profile that may not be the ground state.

    Lastly, if the core value is too small, then the shooting method may get
    stuck because of the large gradients. In this case lower the value of
    radial coordinate spacing, dr.

    Once the generator is complete, the returned object has several methods. for
    example, the profile of the boson star can be called by typing

        boson_star.phi

    """
    def __init__(self,phi0,
                        lamb = 0,
                        alpha_range = [0,1],
                        dr = .01,
                        r_max = 50,
                        phi_tol = 10**-6,
                        printTag = True):
        self.dr = dr
        self.lamb = lamb
        self.generate_profile(phi0,alpha_range,r_max,phi_tol,printTag)
        self.extract_physical_features()
    def extract_physical_features(self):
        #ADM mass definition
        self.mass = .5*self.r*(1-1.0/self.a**2)
        self.M = self.mass[-1]
        self.rho = (.5*(1 + .5*self.lamb*self.phi**2 + 1/self.alpha**2)
                        *self.a*self.phi**2 + .5*self.Phi**2/self.a)
        self.R = (np.trapz(self.a*self.alpha*self.rho*self.r**3,self.r)
                    /np.trapz(self.a*self.alpha*self.rho*self.r**2,self.r))
    def generate_profile(self,phi0,alpha_range,r_max,phi_tol,printTag):
        self.r = [self.dr]
        self.a = [1]
        self.phi = [phi0]
        self.Phi = [0]

        if printTag: plt.clf()

        d_alpha = alpha_range[1] - alpha_range[0]
        while (np.abs(self.phi[-1])>(phi_tol)) and (d_alpha >10**-16):

            self.r = [self.dr]
            self.a = [1]
            self.phi = [phi0]
            self.Phi = [0]

            alpha0 = np.mean(alpha_range)
            self.alpha = [alpha0]

            self._RK4_(r_max)

            #print(self.phi[-5::])
            #print(sum(np.abs(np.diff(np.sign(self.Phi[::-3])))))

            if (np.sign(self.phi[-2]) == 1) and (sum(np.abs(np.diff(np.sign(self.phi[::-3]))))<2):
                alpha_range[1] = alpha_range[1] - .5*d_alpha
            else:
                alpha_range[0] = alpha_range[0] + .5*d_alpha
            d_alpha = .5*d_alpha

            if printTag:
                plt.plot(self.r[:-1],self.phi[:-1],linewidth = .2,color = 'red')
                plt.ylim([-phi0,2*phi0])
                plt.pause(.001)
            #print(str(alpha_range))
        if printTag:
            plt.clf()
            plt.plot(self.r,self.phi)
            plt.pause(.001)

    def _RK4_(self,r_max):
        c = 0.16666666666666666666667

        d       = lambda k1,k2,k3,k4: c*(k1 + 2.0*k2 + 2.0*k3 + k4)
        F_a     = lambda r,a,alpha,phi,Phi: .5*a*( -(a**2-1)/r
                    + r*( (1.0/alpha**2 + 1 + .5*self.lamb*phi**2)*a**2*phi**2
                    + Phi**2))*self.dr
        F_alpha = lambda r,a,alpha,phi,Phi: .5*alpha*( (a**2-1)/r
                    + r*( (1.0/alpha**2 - 1 - .5*self.lamb*phi**2)*a**2*phi**2
                    + Phi**2))*self.dr
        F_phi   = lambda r,a,alpha,phi,Phi: Phi*self.dr
        F_Phi   = lambda r,a,alpha,phi,Phi: (-(1 + a**2 - (a*r*phi)**2)*Phi/r
                    - (1.0/alpha**2 - 1 - self.lamb*phi**2)*a**2*phi)*self.dr

        dr      = self.dr
        r       = self.r
        a       = self.a
        alpha   = self.alpha
        phi     = self.phi
        phi0    = phi[0]
        Phi     = self.Phi

        while ((r[-1] < r_max)
            and (np.abs(a[-1]) < max(100*phi0,100))
            and (np.abs(alpha[-1]) < max(100*phi0,100))
            and (np.abs(phi[-1]) < max(10*phi0,100))
            and (np.abs(Phi[-1]) < max(10*phi0,100))):

            k1_a      = F_a( r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_alpha  = F_alpha( r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_phi    = F_phi( r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])
            k1_Phi    = F_Phi( r[-1], a[-1], alpha[-1], phi[-1], Phi[-1])

            k2_a      = F_a(    r[-1] + .5*dr,
                                a[-1] + .5*k1_a,
                                alpha[-1] + .5*k1_alpha,
                                phi[-1] + .5*k1_phi,
                                Phi[-1] + .5*k1_Phi)
            k2_alpha  = F_alpha(r[-1] + .5*dr,
                                a[-1] + .5*k1_a,
                                alpha[-1] + .5*k1_alpha,
                                phi[-1] + .5*k1_phi,
                                Phi[-1] + .5*k1_Phi)
            k2_phi    = F_phi(  r[-1] + .5*dr,
                                a[-1] + .5*k1_a,
                                alpha[-1] + .5*k1_alpha,
                                phi[-1] + .5*k1_phi,
                                Phi[-1] + .5*k1_Phi)
            k2_Phi    = F_Phi(  r[-1] + .5*dr,
                                a[-1] + .5*k1_a,
                                alpha[-1] + .5*k1_alpha,
                                phi[-1] + .5*k1_phi,
                                Phi[-1] + .5*k1_Phi)
            k3_a      = F_a(    r[-1] + .5*dr,
                                a[-1] + .5*k2_a,
                                alpha[-1] + .5*k2_alpha,
                                phi[-1] + .5*k2_phi,
                                Phi[-1] + .5*k2_Phi)
            k3_alpha  = F_alpha(r[-1] + .5*dr,
                                a[-1] + .5*k2_a,
                                alpha[-1] + .5*k2_alpha,
                                phi[-1] + .5*k2_phi,
                                Phi[-1] + .5*k2_Phi)
            k3_phi    = F_phi(  r[-1] + .5*dr,
                                a[-1] + .5*k2_a,
                                alpha[-1] + .5*k2_alpha,
                                phi[-1] + .5*k2_phi,
                                Phi[-1] + .5*k2_Phi)
            k3_Phi    = F_Phi(  r[-1] + .5*dr,
                                a[-1] + .5*k2_a,
                                alpha[-1] + .5*k2_alpha,
                                phi[-1] + .5*k2_phi,
                                Phi[-1] + .5*k2_Phi)

            k4_a      = F_a(    r[-1] + dr,
                                a[-1] + k3_a,
                                alpha[-1] + k3_alpha,
                                phi[-1] + k3_phi,
                                Phi[-1] + k3_Phi)
            k4_alpha  = F_alpha(r[-1] + dr,
                                a[-1] + k3_a,
                                alpha[-1] + k3_alpha,
                                phi[-1] + k3_phi,
                                Phi[-1] + k3_Phi)
            k4_phi    = F_phi(  r[-1] + dr,
                                a[-1] + k3_a,
                                alpha[-1] + k3_alpha,
                                phi[-1] + k3_phi,
                                Phi[-1] + k3_Phi)
            k4_Phi    = F_Phi(  r[-1] + dr,
                                a[-1] + k3_a,
                                alpha[-1] + k3_alpha,
                                phi[-1] + k3_phi,
                                Phi[-1] + k3_Phi)

            r.append(r[-1] + dr)
            a.append(a[-1] + d(k1_a,k2_a,k3_a,k4_a))
            alpha.append(alpha[-1] + d(k1_alpha,k2_alpha,k3_alpha,k4_alpha))
            phi.append(phi[-1] + d(k1_phi,k2_phi,k3_phi,k4_phi))
            Phi.append(Phi[-1] + d(k1_Phi,k2_Phi,k3_Phi,k4_Phi))

        self.omega = 1.0/(a[-1]*alpha[-1])
        self.r = np.array(r)
        self.a = np.array(a)
        self.alpha = np.array(alpha)*self.omega
        self.phi = np.array(phi)
        self.Phi = np.array(Phi)

