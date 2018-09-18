import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.rcParams["mathtext.fontset"] = "stix"
matplotlib.rcParams['font.family'] = 'STIXGeneral'

class bs1:
    """Boson star class - single scalar field in a harmonic potential"""
    def __init__(self,phi0,alpha_range = [0,.9],dr = .01,r_max = 50):
        self.dr = dr
        self.generate_profile(phi0,alpha_range,r_max)
        self.M = .5*self.r*(1-1.0/self.a**2)
        self.mass = self.M[-1]

    def generate_profile(self,phi0,alpha_range,r_max):
        self.r = [self.dr]
        self.a = [1]
        self.phi = [phi0]
        self.Phi = [0]

        plt.clf()
        while np.abs(self.phi[-1])>(10**-4):
            d_alpha = alpha_range[1] - alpha_range[0]

            self.r = [self.dr]
            self.a = [1]
            self.phi = [phi0]
            self.Phi = [0]

            alpha0 = np.mean(alpha_range)
            self.alpha = [alpha0]

            self._RK4_(r_max)

            if np.sign(self.phi[-1]) == 1:
                alpha_range[1] = alpha_range[1] - .5*d_alpha
            else:
                alpha_range[0] = alpha_range[0] + .5*d_alpha

            plt.plot(self.r[:-1],self.phi[:-1])
            plt.pause(.01)
            print(alpha_range)


    def _RK4_(self,r_max):
        c = 0.16666666666666666666667

        d       = lambda k1,k2,k3,k4: c*(k1 + 2.0*k2 + 2.0*k3 + k4)
        F_a     = lambda r,a,alpha,phi,Phi: .5*a*( -(a**2-1)/r
                    + r*( (1.0/alpha**2 + 1)*a**2*phi**2 + Phi**2))*self.dr
        F_alpha = lambda r,a,alpha,phi,Phi: .5*alpha*( (a**2-1)/r
                    + r*( (1.0/alpha**2 - 1)*a**2*phi**2 + Phi**2))*self.dr
        F_phi   = lambda r,a,alpha,phi,Phi: Phi*self.dr
        F_Phi   = lambda r,a,alpha,phi,Phi: (-(1 + a**2 - (a*r*phi)**2)*Phi/r
                    - (1.0/alpha**2 - 1)*a**2*phi)*self.dr

        dr      = self.dr
        r       = self.r
        a       = self.a
        alpha   = self.alpha
        phi     = self.phi
        phi0 = phi[0]
        Phi     = self.Phi

        while ((r[-1] < r_max)
            and (np.abs(a[-1]) < max(10*phi0,100))
            and (np.abs(alpha[-1]) < max(10*phi0,100))
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

class bs2:
    """Boson star class"""
    def __init__(self,initial_conditions):
        self.dr = .001
        self.phi = [initial_conditions[0]]
        self.Phi = [0]
        self.a = [1]
        self.alpha = [1]
        self.r = [self.dr]

    def plot(self):
        plt.figure(1,figsize = (4,6))
        plt.subplot(311)
        plt.plot(self.r[:-2],self.a[:-2],'-r')
        plt.plot(self.r[:-2],self.alpha[:-2],'-b')
        #plt.xscale('log')
        plt.ylabel('metric')
        plt.legend(['a','$\\alpha$'])
        plt.subplot(312)
        plt.plot(self.r[:-2],self.phi1[:-2],'-r')
        plt.plot(self.r[:-2],self.phi2[:-2],'-b')
        plt.ylabel('momenta')
        #plt.xscale('log')
        plt.legend(['$\Phi_2$','$\Phi_2$'])
        plt.subplot(313)
        plt.plot(self.r[:-2],self.psi1[:-2],'-r')
        plt.plot(self.r[:-2],self.psi2[:-2],'-b')
        plt.xlabel('$r$')
        #plt.xscale('log')
        plt.ylabel('fields')
        plt.legend(['$\psi_1$','$\psi_2$'])

    def _RK4(self,omegas,potential_params,phi_max = 100,psi_max = 100,met_max = 100):
        #Extracts the input parameters
        mu,lambda1,lambda2,lambda12 = potential_params
        omega1,omega2 = omegas
        #Define all the equations of motion and the potential
        U = lambda psi1,psi2:(-mu**2*psi1**2 + .5*lambda1*psi1**4
                + .5*lambda2*psi2**4 + lambda12*psi1**2*psi2**2
                + .5*mu**4/lambda1)
        Fa = lambda r,a,alpha,phi1,phi2,psi1,psi2: a*((1-a**2)/(2.0*r)
                + 4*np.pi*r*(phi1**2 + phi2**2
                + (a/alpha)**2*(omega1**2*psi1**2 + omega2**2*psi2**2)
                + a**2*U(psi1,psi2)
                ))
        Fb = lambda r,a,alpha,phi1,phi2,psi1,psi2: alpha*((a**2-1)/(2.0*r)
                + 4*np.pi*r*(phi1**2 + phi2**2
                + (a/alpha)**2*(omega1**2*psi1**2 + omega2**2*psi2**2)
                - a**2*U(psi1,psi2)
                ))
        FP1 = lambda r,a,alpha,phi1,phi2,psi1,psi2: ((-(1+a**2)/r
                + 8*np.pi*r*a**2*U(psi1,psi2))*phi1 + a**2*(
                -(mu**2 + (omega1/alpha)**2)*psi1 + lambda1*psi1**3
                + lambda12*psi1*psi2**2
                ))
        FP2 = lambda r,a,alpha,phi1,phi2,psi1,psi2: ((-(1+a**2)/r
                + 8*np.pi*r*a**2*U(psi1,psi2))*phi2 - a**2*(
                -(omega2/alpha)**2*psi2 + lambda2*psi2**3
                + lambda12*psi1**2*psi2
                ))
        Fp1 = lambda r,a,alpha,phi1,phi2,psi1,psi2: phi1
        Fp2 = lambda r,a,alpha,phi1,phi2,psi1,psi2: phi2

        while ((self.r[-1] < 20) and (np.abs(self.psi1[-1])<psi_max)
            and (np.abs(self.psi2[-1])<psi_max) and (np.abs(self.phi1[-1])<phi_max)
            and (np.abs(self.phi2[-1])<phi_max) and (np.abs(self.a[-1])<met_max)
            and (np.abs(self.alpha[-1])<met_max)):

            ka1 = Fa(self.r[-1],self.a[-1],self.alpha[-1],self.phi1[-1],
                        self.phi2[-1],self.psi1[-1],self.psi2[-1])*self.dr
            kb1 = Fb(self.r[-1],self.a[-1],self.alpha[-1],self.phi1[-1],
                        self.phi2[-1],self.psi1[-1],self.psi2[-1])*self.dr
            kP11 = FP1(self.r[-1],self.a[-1],self.alpha[-1],self.phi1[-1],
                        self.phi2[-1],self.psi1[-1],self.psi2[-1])*self.dr
            kP21 = FP2(self.r[-1],self.a[-1],self.alpha[-1],self.phi1[-1],
                        self.phi2[-1],self.psi1[-1],self.psi2[-1])*self.dr
            kp11 = Fp1(self.r[-1],self.a[-1],self.alpha[-1],self.phi1[-1],
                        self.phi2[-1],self.psi1[-1],self.psi2[-1])*self.dr
            kp21 = Fp2(self.r[-1],self.a[-1],self.alpha[-1],self.phi1[-1],
                        self.phi2[-1],self.psi1[-1],self.psi2[-1])*self.dr

            ka2 = Fa(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka1,
                     self.alpha[-1] + .5*kb1,
                     self.phi1[-1] + .5*kP11,
                     self.phi2[-1] + .5*kP21,
                     self.psi1[-1] + .5*kp11,
                     self.psi2[-1] + .5*kp21)*self.dr
            kb2 = Fb(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka1,
                     self.alpha[-1] + .5*kb1,
                     self.phi1[-1] + .5*kP11,
                     self.phi2[-1] + .5*kP21,
                     self.psi1[-1] + .5*kp11,
                     self.psi2[-1] + .5*kp21)*self.dr
            kP12 = FP1(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka1,
                     self.alpha[-1] + .5*kb1,
                     self.phi1[-1] + .5*kP11,
                     self.phi2[-1] + .5*kP21,
                     self.psi1[-1] + .5*kp11,
                     self.psi2[-1] + .5*kp21)*self.dr
            kP22 = FP2(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka1,
                     self.alpha[-1] + .5*kb1,
                     self.phi1[-1] + .5*kP11,
                     self.phi2[-1] + .5*kP21,
                     self.psi1[-1] + .5*kp11,
                     self.psi2[-1] + .5*kp21)*self.dr
            kp12 = Fp1(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka1,
                     self.alpha[-1] + .5*kb1,
                     self.phi1[-1] + .5*kP11,
                     self.phi2[-1] + .5*kP21,
                     self.psi1[-1] + .5*kp11,
                     self.psi2[-1] + .5*kp21)*self.dr
            kp22 = Fp2(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka1,
                     self.alpha[-1] + .5*kb1,
                     self.phi1[-1] + .5*kP11,
                     self.phi2[-1] + .5*kP21,
                     self.psi1[-1] + .5*kp11,
                     self.psi2[-1] + .5*kp21)*self.dr

            ka3 = Fa(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka2,
                     self.alpha[-1] + .5*kb2,
                     self.phi1[-1] + .5*kP12,
                     self.phi2[-1] + .5*kP22,
                     self.psi1[-1] + .5*kp12,
                     self.psi2[-1] + .5*kp22)*self.dr
            kb3 = Fb(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka2,
                     self.alpha[-1] + .5*kb2,
                     self.phi1[-1] + .5*kP12,
                     self.phi2[-1] + .5*kP22,
                     self.psi1[-1] + .5*kp12,
                     self.psi2[-1] + .5*kp22)*self.dr
            kP13 = FP1(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka2,
                     self.alpha[-1] + .5*kb2,
                     self.phi1[-1] + .5*kP12,
                     self.phi2[-1] + .5*kP22,
                     self.psi1[-1] + .5*kp12,
                     self.psi2[-1] + .5*kp22)*self.dr
            kP23 = FP2(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka2,
                     self.alpha[-1] + .5*kb2,
                     self.phi1[-1] + .5*kP12,
                     self.phi2[-1] + .5*kP22,
                     self.psi1[-1] + .5*kp12,
                     self.psi2[-1] + .5*kp22)*self.dr
            kp13 = Fp1(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka2,
                     self.alpha[-1] + .5*kb2,
                     self.phi1[-1] + .5*kP12,
                     self.phi2[-1] + .5*kP22,
                     self.psi1[-1] + .5*kp12,
                     self.psi2[-1] + .5*kp22)*self.dr
            kp23 = Fp2(self.r[-1] + .5*self.dr,
                     self.a[-1] + .5*ka2,
                     self.alpha[-1] + .5*kb2,
                     self.phi1[-1] + .5*kP12,
                     self.phi2[-1] + .5*kP22,
                     self.psi1[-1] + .5*kp12,
                     self.psi2[-1] + .5*kp22)*self.dr

            ka4 = Fa(self.r[-1] + self.dr,
                     self.a[-1] + ka3,
                     self.alpha[-1] + kb3,
                     self.phi1[-1] + kP13,
                     self.phi2[-1] + kP23,
                     self.psi1[-1] + kp13,
                     self.psi2[-1] + kp23)*self.dr
            kb4 = Fb(self.r[-1] + self.dr,
                     self.a[-1] + ka3,
                     self.alpha[-1] + kb3,
                     self.phi1[-1] + kP13,
                     self.phi2[-1] + kP23,
                     self.psi1[-1] + kp13,
                     self.psi2[-1] + kp23)*self.dr
            kP14 = FP1(self.r[-1] + self.dr,
                     self.a[-1] + ka3,
                     self.alpha[-1] + kb3,
                     self.phi1[-1] + kP13,
                     self.phi2[-1] + kP23,
                     self.psi1[-1] + kp13,
                     self.psi2[-1] + kp23)*self.dr
            kP24 = FP2(self.r[-1] + self.dr,
                     self.a[-1] + ka3,
                     self.alpha[-1] + kb3,
                     self.phi1[-1] + kP13,
                     self.phi2[-1] + kP23,
                     self.psi1[-1] + kp13,
                     self.psi2[-1] + kp23)*self.dr
            kp14 = Fp1(self.r[-1] + self.dr,
                     self.a[-1] + ka3,
                     self.alpha[-1] + kb3,
                     self.phi1[-1] + kP13,
                     self.phi2[-1] + kP23,
                     self.psi1[-1] + kp13,
                     self.psi2[-1] + kp23)*self.dr
            kp24 = Fp2(self.r[-1] + self.dr,
                     self.a[-1] + ka3,
                     self.alpha[-1] + kb3,
                     self.phi1[-1] + kP13,
                     self.phi2[-1] + kP23,
                     self.psi1[-1] + kp13,
                     self.psi2[-1] + kp23)*self.dr

            self.r.append(self.r[-1] + self.dr)
            self.a.append(self.a[-1] + .166666666667*(ka1 + 2.0*ka2 + 2.0*ka3 + ka4))
            self.alpha.append(self.alpha[-1] + .166666666667*(kb1 + 2.0*kb2 + 2.0*kb3 + kb4))
            self.phi1.append(self.phi1[-1] + .166666666667*(kP11 + 2.0*kP12 + 2.0*kP13 + kP14))
            self.phi2.append(self.phi2[-1] + .166666666667*(kP21 + 2.0*kP22 + 2.0*kP23 + kP24))
            self.psi1.append(self.psi1[-1] + .166666666667*(kp11 + 2.0*kp12 + 2.0*kp13 + kp14))
            self.psi2.append(self.psi2[-1] + .166666666667*(kp21 + 2.0*kp22 + 2.0*kp23 + kp24))
