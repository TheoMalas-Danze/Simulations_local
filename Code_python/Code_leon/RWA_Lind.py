# This code implements a class that performs RWA computations and computes effective Lindbladians
# Modified 10/11 added a degree function to the class Expression does not work with sin and cos
#Modified 21/12 added the handling of the expansion corrected mistake in moyal_product method

import sympy as sp
import numpy as np
import scipy
from IPython.display import display, Math, Latex


class System:
    def __init__(self, dim, n_osc: int):
        """
        Structure: H_0 has shape [dim[oscillator_i],dim[oscillator_i]  for i in n_osc]
                morally H_0 has 2* n_osc entries for the annihilation and creation op of each oscillator
        The entry H[0,1] contains the prefactor of the term \hat{a_0} in the Hamiltonian for instance
        :param dim: a list of size n_osc with the dimension of each osc or an interger
        :param n_osc: number of oscillators
        """
        assert type(dim) == int, "dim should be an integer"
        self.n_osc = n_osc  # Number of modes
        self.dim = dim
        # A dummy array to quickly initialize arrays of the right size
        self.H_0 = np.zeros([dim for i in range(2 * self.n_osc)])

    def make_structure_cste_ho(self):
        """
        Returns the contribution to a given monome(order) to the commutator of two monomes.
        It is computed using the one mode case (n_osc=1) and extended by using the following relation:
            [(l1,m1)(l2,m2),(k1,n1)(k2,n2)] = [(l1,m1),(k1,n1)] [(l2,m2),(k2,n2)]
                                                + [(l1,m1),(k1,n1)] (k2,n2)*(l2,m2)
                                                + (k1,n1)*(l1,m1) [(l2,m2),(k2,n2)]
        instancates a function that takes a list of indices and returns the structure constant
        """
        def st_cste_1ho(l: int, m: int, k: int, n: int, i: int, j: int):
            """ coefficient (i,j) in the commutator [(l,m),(k,n)]  for a 1 mode system"""
            if k + l - i == m + n - j > 0:
                u = k + l - i
                return scipy.special.factorial(k + l - i) * (scipy.special.binom(k, u) * scipy.special.binom(m, u)
                                                             - scipy.special.binom(l, u) * scipy.special.binom(n, u))
            else:
                return 0

        def normal_ordering(l: int, m: int, k: int, n: int, i: int, j: int):
            """ coefficient (i,j) in the product (l,m)*(k,n) """
            if i == l + k and j == m + n:
                return 1
            elif i < l or j < n:
                return 0
            else:
                return st_cste_1ho(0, m, k, 0, i - l, j - n)

        def custom_1ho(select: bool, l: int, m: int, k: int, n: int, i: int, j: int):
            """ wrapping function of the two previous ones"""
            if select:
                # Commutator [(l,m),(k,n)]
                return st_cste_1ho(l, m, k, n, i, j)
            else:
                return normal_ordering(k, n, l, m, i, j)  # product (k,n)*(l,m)

        def st_cste(Indices: np.ndarray):
            """
            Returns the structure constant given the list of indices it is computed using the one mode case
            and the following relation:
            [(l1,m1)(l2,m2),(k1,n1)(k2,n2)] = [(l1,m1),(k1,n1)] [(l2,m2),(k2,n2)]
                                                + [(l1,m1),(k1,n1)] (k2,n2)*(l2,m2)
                                                +(k1,n1)*(l1,m1) [(l2,m2),(k2,n2)]
            :param Indices: a list of 6 indices (l1,m1,k1,n1,l2,m2)
            :return: the structure constant
            """
            assert len(
                Indices) == 6 * self.n_osc, f"Wrong number of indices expected {6 * self.n_osc} got {len(Indices.shape)}"
            C = 0
            # This deals with the cross-terms
            for i in range(1, 2 ** self.n_osc):
                # Fancy way of alternating between 1 mode cummutator and normal ordering
                a = np.binary_repr(i, self.n_osc)
                C += np.prod(
                    [custom_1ho(bool(int(a[j])),
                                *Indices[[2 * j, 2 * j + 1, 2 * self.n_osc +
                                          2 * j, 2 * self.n_osc + 2 * j + 1]],
                                *Indices[[4 * self.n_osc + 2 * j, 4 * self.n_osc + 2 * j + 1]]) for j in
                     range(self.n_osc)])
            return C
        self.st_cste = st_cste
        self.normal_1ho = normal_ordering

    def commute_custom(self, o1, o2):
        """
        Computes the commutator of two operators in normal order
        :param o1: operator 1
        :param o2: operator 2
        :return: commutator
        """
        o3 = 0*o1.copy()
        I1 = np.array(np.where(o1 != 0))
        I2 = np.array(np.where(o2 != 0))
        key_max = np.max(I1, 1)+np.max(I2, 1)
        """ for key3, value3 in np.ndenumerate(o3):
            if np.sum(np.array(key3)<=key_max)==2*self.n_osc:
                o3[key3] = np.sum(
                    [self.st_cste(np.array(key1 + key2 + key3)) * value1 * value2 for key1, value1 in np.ndenumerate(o1) for
                    key2, value2 in np.ndenumerate(o2) if value1!=0 and value2!=0]) """
        for key3, value3 in np.ndenumerate(o3):
            if np.sum(np.array(key3) <= key_max) == 2*self.n_osc:
                for key1 in I1.transpose():
                    for key2 in I2.transpose():
                        o3[key3] += self.st_cste(np.array(tuple(key1) + tuple(
                            key2) + key3)) * o1[tuple(key1)] * o2[tuple(key2)]
        return o3

    def prod_custom(self, o1, o2):
        """
        Computes the product (l1,m1)(l2,m2)*(k1,n1)(k2,n2) = ((l1,m1)*(k1,n1))((l2,m2)*(k2,n2))
        in normal order
        :param o1: operator 1
        :param o2: operator 2
        :return: product of o1 and o2
        """
        o3 = 0*o1.copy()
        I1 = np.array(np.where(o1 != 0))
        I2 = np.array(np.where(o2 != 0))
        # This is the maximum key that can be reached
        key_max = np.max(I1, 1)+np.max(I2, 1)
        for key3, value3 in np.ndenumerate(o3):
            if np.sum(np.array(key3) <= key_max) == 2*self.n_osc:
                for key1 in I1.transpose():
                    for key2 in I2.transpose():
                        o3[key3] += np.prod([self.normal_1ho(*np.array(key1)[2*i:2*(i+1)], *np.array(key2)[2*i:2*(
                            i+1)], *np.array(key3)[2*i:2*(i+1)]) for i in range(self.n_osc)])*o1[tuple(key1)]*o2[tuple(key2)]
        return o3

    def make_ops(self):
        """
        This function creates the operators a and a^dagger in the form of a 2n dimensional array
        """
        def make_array(args):
            assert len(args) == 2*self.n_osc, 'wrong number of entries'
            A = np.zeros([self.dim for i in range(2*self.n_osc)])
            A[args] = 1
            return A
        self.ladder = make_array

    def formating_rotating(self, O: np.ndarray):
        """
        This function takes an operator O and returns the same operator formated such that 
        it takes the form: O = sum_{i} O_i e^{sp.I (F(omega)) st}
        In particular it ensures that the exponentials are grouped together at the numerator and simplified.
        :param O: operator
        :return: formated operator
        """
        O_copy = np.copy(O)
        I = np.array(np.argwhere(O_copy != 0))
        for i in I:
            O_copy[tuple(i)] = sp.expand(O_copy[tuple(i)], power_exp=False)
            Expr2 = 0*sp.I
            # If O is a product there is no need to do anything
            if type(O_copy[tuple(i)]) == sp.core.add.Add:
                for a in O_copy[tuple(i)].args:
                    Expr2 += sp.powsimp(sp.factor_terms(a), force=True)
                O_copy[tuple(i)] = Expr2
        return O_copy

    def make_Hsb(self, H: np.ndarray):
        """
        This function defines the system-bath Hamiltonian
        :param H: system-bath Hamiltonian
        """
        assert len(H.shape) == 2 * \
            self.n_osc, f"Wrong shape expected {2 * self.n_osc} got {len(H.shape)}"
        self.Hsb = H

    def make_Ht(self, H: np.ndarray, tim=sp.symbols('t', positive=True)):
        """
        This function defines the Hamiltonian of the system in the rotating frame
        :param H: Hamiltonian of the system
        :param tim: time sympy variable useful for the RWA
        """
        assert len(H.shape) == 2 * \
            self.n_osc, f"Wrong shape expected {2 * self.n_osc} got {len(H.shape)}"
        self.Ht = self.formating_rotating(H)
        self.t = tim

    def not_collect(self, expr: sp.Expr, x: sp.Expr):
        """
        Returns the terms that do not depend on x in the expression. The expression should be in an expanded form.
        :param expr: expression
        :param x: variable
        """
        exp2 = 0
        exps = expr + 1  # ensure exps is of type Add even if expr is a monomial
        # iterate over the terms of the expression (which is an Add object/a sum)
        for e in exps.args:
            if e.has(x):
                e = 0
            exp2 += e
        return exp2-1

    def get_average(self, O):
        """
        UNUSED
        Returns the average of the operator O wrt to time and the oscillatory part
        :param O: operator in the rotating format
        :return: average, oscillatory part
        """
        Ob, Oosc = np.copy(O)*0, np.copy(O)*0
        I = np.array(np.argwhere(O != 0))
        for i in I:
            key = tuple(i)
            Ob[key] = self.not_collect(self.O[key], sp.exp)
            Oosc[key] = 1*self.O[key] - Ob[key]
            Oosc[key] = sp.expand(Oosc[key], power_exp=False)
            Expr2 = 0*sp.I
            if type(Oosc[key]) == sp.core.add.Add:
                for a in Oosc[key].args:
                    Expr2 += sp.powsimp(sp.factor_terms(a), force=True)
                Oosc[key] = Expr2
        return Ob, Oosc

    def get_ampfreq(self, expr):
        """
        TODO: simplify this function
        This function returns the amplitudes and frequencies of the expression of the type sum_i A_i e^{sp.I F_i t}
        It is useful to obtain the effective dissipators at each frequency.
        This fuction assumes that expr is a nested expresion of sum and products (no exponential should be at the denominator)
        :param expr: expression
        :return: amplitudes, frequencies
        """
        se = sp.symbols('e', positive=True)
        A, F = [], []
        if type(expr) == sp.core.mul.Mul:  # if the expression is first a multiplication
            prefactor = 1
            freq_shift = 0
            AV, FV = [], []
            for e in expr.args:
                if not e.has(sp.exp):  # e is a prefactor
                    prefactor *= e
                elif type(e) != sp.core.add.Add:  # multiplicative exponential
                    freq_shift += e.subs(sp.exp,
                                         se).as_coeff_exponent(se)[-1]/(sp.I*self.t)
                else:  # Sum with exponentials
                    A2, F2 = self.get_ampfreq(e)
                    for key, value in np.ndenumerate(F2):
                        if not value in FV:
                            FV.append(F2[key[0]])
                            AV.append(A2[key[0]])
                        elif value in FV:
                            i = int(np.argwhere(np.array(FV) == value))
                            AV[i] += A2[key[0]]
                        else:
                            raise('value neither in or not in list')
            if len(FV) == 0:  # if there is no exponential in the expression
                AV = [1]
                FV = [0]
            F += [FV[i] + freq_shift for i in range(len(FV))]
            A += [AV[i]*prefactor for i in range(len(AV))]
        elif type(expr) == sp.core.add.Add:  # expr is first a sum
            for e in expr.args:
                if e.has(sp.exp):  # the term has an exponential
                    amplitude, freq = (e.subs(sp.exp, se)
                                       ).as_coeff_exponent(se)
                    freq = freq/(sp.I*self.t)
                    if freq != 0:
                        if freq in F:
                            i = int(np.argwhere(np.array(F) == freq))
                            A[i] += amplitude
                        elif freq not in F:
                            F.append(freq)
                            A.append(amplitude)
                    else:  # nested sum a*b*(C+d*(U e^() + V )) or constant
                        A2, F2 = self.get_ampfreq(e)
                        for key, value in np.ndenumerate(F2):
                            if not value in F:
                                F.append(F2[key[0]])
                                A.append(A2[key[0]])
                            elif value in F:
                                i = int(np.argwhere(np.array(F) == value))
                                A[i] += A2[key[0]]
                            else:
                                raise('value neither in or not in list')
                else:                               # Constants
                    if 0 in F:
                        i = int(np.argwhere(np.array(F) == 0))
                        A[i] += e
                    else:
                        A.append(e)
                        F.append(0)
        else:  # expr is only an exponential/a constant
            if expr.has(sp.exp):
                amplitude, freq = (expr.subs(sp.exp, se)).as_coeff_exponent(se)
                A = [amplitude]
                F = [freq]
            else:
                A = [expr]
                F = [0]
        return A, F

    def RWA_1(self):
        """
        Performs first order RWA. 
        Instanciates the effective hamiltonian, the oscillatory part, the generator and the System-bath hamiltonian
        """
        I = np.array(np.argwhere(self.Ht != 0))
        self.Hb, self.Hosc, self.S1 = np.copy(
            self.Ht)*0, np.copy(self.Ht)*0, np.copy(self.Ht)*0
        for i in I:
            key = tuple(i)
            # selects the terms involving no exponential (Caution: do not include exp that are time indep)
            self.Hb[key] = self.not_collect(self.Ht[key], sp.exp)
            if sum(i) != 0:  # constant terms are going to disappear in the commutator
                self.Hosc[key] = self.Ht[key] - self.Hb[key]
                self.Hosc[key] = sp.expand(self.Hosc[key], power_exp=False)
                # Same as formating_rotating method, however it is more efficient to do it here
                Expr2 = 0*self.t
                if type(self.Hosc[key]) == sp.core.add.Add:
                    for a in self.Hosc[key].args:
                        Expr2 += sp.powsimp(sp.factor_terms(a), force=True)
                    self.Hosc[key] = Expr2
            # integration
            self.S1[key] = - \
                self.Hosc[key].replace(
                    sp.exp, lambda x: sp.exp(x)/x.subs(self.t, 1))
        # Handle System-bath interaction
        self.Hsb_1 = (-sp.I)*self.commute_custom(self.S1, self.Hsb)
        self.Hsb_1 = self.formating_rotating(self.Hsb_1)

    def RWA_2(self):
        """
        Performs second order RWA. 
        Instanciates the effective hamiltonian, the oscillatory part, the generator and the System-bath hamiltonian
        """
        self.Hb2, self.Hosc2, self.S2 = np.copy(
            self.Ht)*0, np.copy(self.Ht)*0, np.copy(self.Ht)*0
        dS2 = (-sp.I) * self.commute_custom(self.S1, self.Hb + 0.5*self.Hosc)
        dS2 = self.formating_rotating(dS2)
        I = np.array(np.argwhere(dS2 != 0))
        for i in I:
            key = tuple(i)
            self.Hb2[key] = self.not_collect(dS2[key], sp.exp)
            if sum(i) != 0:  # constant terms are going to disappear in the commutator
                self.Hosc2[key] = dS2[key] - self.Hb2[key]
                self.Hosc2[key] = sp.expand(self.Hosc2[key], power_exp=False)
                # Same as formating_rotating method, however it is more efficient to do it here
                Expr2 = 0*self.t
                if type(self.Hosc2[key]) == sp.core.add.Add:
                    for a in self.Hosc2[key].args:
                        Expr2 += sp.powsimp(sp.factor_terms(a), force=True)
                    self.Hosc2[key] = Expr2
            # integration
            self.S2[key] = - \
                self.Hosc2[key].replace(
                    sp.exp, lambda x: sp.exp(x)/x.subs(self.t, 1))
        # Handle System-bath interaction
        self.Hsb_2 = (-sp.I)*self.commute_custom(self.S2, self.Hsb)+(-sp.I)/2 * \
            self.commute_custom(
                self.S1, self.Hsb_1)  # ,self.commute_custom(self.S1,self.Hsb))
        self.Hsb_2 = self.formating_rotating(self.Hsb_2)

    def RWA_3(self):
        """
        Performs third order RWA. 
        Instanciates the effective hamiltonian, the oscillatory part, the generator and the System-bath hamiltonian
        """
        self.Hb3, self.Hosc3, self.S3 = np.copy(
            self.Ht)*0, np.copy(self.Ht)*0, np.copy(self.Ht)*0
        dS3 = ((-sp.I) * self.commute_custom(self.S2, self.Hb + 0.5*self.Hosc)
               + (-sp.I)/2 * self.commute_custom(self.S1, -self.Hosc2 + (-sp.I)*self.commute_custom(
                   self.S1, self.Hb + 2/3*self.Hosc)))
        dS3 = self.formating_rotating(dS3)
        I = np.array(np.argwhere(dS3 != 0))
        for i in I:
            key = tuple(i)
            self.Hb3[key] = self.not_collect(dS3[key], sp.exp)
            if sum(i) != 0:  # constant terms are going to disappear in the commutator
                self.Hosc3[key] = dS3[key] - self.Hb3[key]
                self.Hosc3[key] = sp.expand(self.Hosc3[key], power_exp=False)
                # Same as formating_rotating method, however it is more efficient to do it here
                Expr2 = 0*self.t
                if type(self.Hosc3[key]) == sp.core.add.Add:
                    for a in self.Hosc3[key].args:
                        Expr2 += sp.powsimp(sp.factor_terms(a), force=True)
                    self.Hosc3[key] = Expr2
            # integration
            self.S3[key] = - \
                self.Hosc3[key].replace(
                    sp.exp, lambda x: sp.exp(x)/x.subs(self.t, 1))
        # Handle System-bath interaction
        self.Hsb_3 = ((-sp.I)*self.commute_custom(self.S3, self.Hsb)
                      + (-sp.I)/2*self.commute_custom(self.S2, self.Hsb_1) +
                      (-sp.I)/2*self.commute_custom(self.S1, self.Hsb_2)
                      - (-sp.I)**2/12*self.commute_custom(self.S1, self.commute_custom(self.S1, self.Hsb_1)))
        self.Hsb_3 = self.formating_rotating(self.Hsb_3)

    def freq_dissipators(self, H_pert: np.ndarray):
        """
        Computes the dissipators coming from H_pert. In H_pert it groups terms according to their exponents 
        and computes the corresponding dissipators.
        :param H_pert: Total system-bath hamiltonian Hsb + Hsb_1 + Hsb_2 + ...
        :return: Freqs: dictionary of dissipators indexed by their frequency 
        """
        Freqs = {}
        I = np.array(np.argwhere(H_pert != 0))
        for ikey in I:
            key = tuple(ikey)
            A, F = self.get_ampfreq(H_pert[key])
            for keyf, valuef in np.ndenumerate(F):
                if valuef in Freqs:
                    Freqs[valuef].append([key, A[keyf[0]]])
                else:
                    Freqs[valuef] = [[key, A[keyf[0]]]]
        return Freqs

    def print_operator(self, O):
        """
        Prints an operator in terms of the creation and annihilation operators
        :param O: operator to be printed
        """
        a = [sp.symbols('\hat{a}_'+f'{i}', commutative=False)
             for i in range(self.n_osc)]
        ad = [sp.symbols('\hat{a}^\dagger_'+f'{i}', commutative=False)
              for i in range(self.n_osc)]
        expr = 0*sp.I
        I = np.array(np.argwhere(O != 0))
        for i in I:
            expr += np.prod([ad[j]**i[j*2] * a[j]**i[j*2 + 1]
                             for j in range(self.n_osc)]) * O[tuple(i)]
        return expr

    def print_dissipator(self, w, Freq):
        """
        Prints the dissipator corresponding to frequency w
        :param w: frequency of the dissipator
        :param Freq: dictionary of dissipators indexed by their frequency
        """
        a = [sp.symbols('\hat{a}_'+f'{i}', commutative=False)
             for i in range(self.n_osc)]
        ad = [sp.symbols('\hat{a}^\dagger_'+f'{i}', commutative=False)
              for i in range(self.n_osc)]
        expr = 0*sp.I
        for L in Freq[w]:
            if L[0][0] == 0 and L[0][1] == 0 and L[0][2] == 0 and L[0][3] == 0:
                continue
            # ad**L[0][0] * a**L[0][1] * bd**L[0][2] * b**L[0][3] *L[1]
            expr += np.prod([ad[j]**L[0][j*2] * a[j]**L[0][j*2 + 1]
                             for j in range(self.n_osc)]) * L[1]
        return expr

# This is a Phase-space version of the above

class Expression:

    def __init__(self, hbar=1) -> None:
        """
        :param hbar: hbar is the value of [a,a^dagger] or [q,p] or [phi,n]
        """
        self.hbar = hbar
        self.t = sp.symbols('t', real=True)
        self.H0 = 0*self.t

    def not_collect(self, expr: sp.Expr, x: sp.Expr):
        """
        Returns the terms that do not depend on x in the expr that should be in an expanded form.
        :param expr: expression
        :param x: variable
        """
        exp2 = 0
        exps = expr + 1  # ensures exps is of type Add even if expr is a monomial
        for e in exps.args:
            if e.has(x):
                e = 0
            exp2 += e
        return exp2-1

    def formating_rotating(self, O: sp.Expr):
        """
        This function takes an operator O and returns the same operator formated such that 
        it takes the form: O = sum_{i} O_i e^{sp.I (F(omega)) st}
        In particular it ensures that the exponentials are grouped together at the numerator and simplified.
        An optimal use of this method would be to perform the integration along with the formating.
        :param O: operator
        :return: formated operator
        """
        O_copy = 1 * O +1
        O_copy = sp.expand(O_copy, power_exp=False)
        Expr2 = 0*sp.I
        if type(O_copy) == sp.core.add.Add:
            for a in O_copy.args:
                Expr2 += sp.powsimp(sp.factor_terms(a), force=True)
            O_copy = Expr2
        return O_copy-1
        #return sp.powsimp(sp.expand(O,power_exp=False),force=True)

    def delete_constants(self, exp):
        """
        Deletes the constants wrt the operators a, a^dagger 
        :param exp: expression should be of type Add
        :return: expression without constants
        """
        if exp==0:
            return exp
        u = exp.expand() + 1
        for a in u.args:
            if not a.has(*self.a, *self.ad):
                u -= a
        return u

    def make_order_param(self, o_param:sp.Expr):
        """
        This function defines the order parameter of the system
        :param o_param: order parameter
        """
        self.order_param = o_param

    def make_ladders(self, number_oscillators):
        """
        Define the operators a_i and a_i^dagger for 1<=i<=number_oscillators
        :param number_oscillators: number of oscillators
        """
        self.n_osc = number_oscillators
        self.a = [sp.Symbol(f'a_{i}') for i in range(number_oscillators)]
        self.ad = [sp.Symbol(f'a_{i}^\dagger')
                   for i in range(number_oscillators)]
        
    def make_Hsb(self, Hsb: sp.Expr):
        """
        The expression of H0 should be in normal order
        :param H0: the hamiltonian of the system using the operators a_i and a_i^dagger
        """
        self.Hsb = self.formating_rotating(Hsb)

    def make_Ht(self, Ht: sp.Expr, tim=sp.symbols('t', positive=True)):
        """
        The expression of Ht should be in normal order
        :param Ht: the hamiltonian of the system using the operators a_i and a_i^dagger
        :param tim: time sympy variable useful for the RWA
        """
        self.Ht = self.formating_rotating(Ht)
        self.t = tim

    def displacement(self, iosc:int, alpha:sp.Expr, sO:sp.Expr, trunc=10):
        """
        Performs the displacement on Operator sO. Defined by 
        D(\alpha ) = \exp \left(\alpha a^\dagger - \alpha ^\ast a\right)
        :param iosc: index of the mode or the mode annihilation operator
        :param alpha: time function of the displacement, must be expressed exclusively with e^{it \omega's}
        :param sO: operator to be displaced
        :return: displaced operator
        """
        X = -alpha * self.ad[iosc]+ sp.conjugate(alpha) * self.a[iosc]
        sOc = sO + 1
        sO2 = 0
        for i in range(len(sOc.args)):
            H_com = [sOc.args[i]*1]
            if H_com[-1].has(self.a[iosc]) or H_com[-1].has(self.ad[iosc]):
                for i in range(1,trunc+1):
                    H_com.append(self.hussimi_bracket(X, H_com[-1])*sp.I)
                    if H_com[-1] == 0:
                        break
                if H_com[-1] != 0: 
                    print('truncated BCH formula at order',trunc)
            sO2 += sum([H_com[i]/sp.factorial(i) for i in range(len(H_com))])
        sO2 += -sp.I * alpha.diff(self.t) * self.ad[iosc] + sp.I* sp.conjugate(alpha.diff(self.t)) * self.a[iosc]
        return sO2-1
    
    def displace_cl_sol(self,iosc:int):
        """ 
        Displace the Hamiltonian by the classical solutions of a off-resonant drive
        The Hamiltonian should be instanciated
        :param iosc: index of the mode or the mode annihilation operator
        """
        assert hasattr(self,'Ht'), 'Ht not defined use make_Ht'
        a_i, ad_i = self.a[iosc], self.ad[iosc]
        Lin_term = self.Ht.coeff(ad_i).subs({self.a[i]:0 for i in range(self.n_osc)}).subs({self.ad[i]:0 for i in range(self.n_osc)})
        if Lin_term == 0:
            print('Classical solution is 0')
        else:
            D = self.Ht.coeff(a_i*ad_i).subs({self.a[i]:0 for i in range(self.n_osc)}).subs({self.ad[i]:0 for i in range(self.n_osc)})
            if D.has(self.t):
                print('No constant detuning')
            else:
                A,f = self.get_ampfreq(Lin_term)
                alpha = sum([-A[i]*sp.exp(sp.I*self.t*f[i])/(D+f[i]) for i in range(len(A))])
                self.Ht = self.formating_rotating(self.delete_constants(self.displacement(iosc, alpha, self.Ht)))
    
    def simplify_osc(self, expr:sp.Expr, iosc:int):
        """
        Simplifies the expression expr order by order in the annihilation operators a_i
        :param expr: expression to be simplified
        :param iosc: index of the mode or the mode annihilation operator
        :return: simplified expression
        """
        order = max(sp.degree(expr, self.a[iosc]), sp.degree(expr, self.ad[iosc]))
        Args = list((expr+1).args)
        for i in range(1,order+1):
            for j in range(1,order+1):
                indj = np.argwhere([a.diff(self.a[iosc]**i*self.ad[iosc]**j)\
                    .subs({self.a[k]:0 for k in range(len(self.a))}).subs({self.ad[k]:0 for k in range(len(self.a))})!=0 for a in Args])
                Asum = sum([Args[ind[0]] for ind in indj]).simplify()
                Args = [Args[ind] for ind in range(len(Args)) if ind not in indj[:,0]] + [Asum] 
        return sum(Args)-1

    def get_ampfreq(self, expr):
        """
        TODO: simplify this function
        This function returns the amplitudes and frequencies of the expression of the type sum_i A_i e^{sp.I F_i t}
        It is useful to obtain the effective dissipators at each frequency.
        This fuction assumes that expr is a nested expresion of sum and products (no exponential should be at the denominator)
        :param expr: expression
        :return: amplitudes, frequencies
        """
        se = sp.symbols('e', positive=True)
        A, F = [], []
        if type(expr) == sp.core.mul.Mul:  # if the expression is first a multiplication
            prefactor = 1
            freq_shift = 0
            AV, FV = [], []
            for e in expr.args:
                if not e.has(sp.exp):  # e is a prefactor
                    prefactor *= e
                elif type(e) != sp.core.add.Add:  # multiplicative exponential
                    freq_shift += e.subs(sp.exp,
                                         se).as_coeff_exponent(se)[-1]/(sp.I*self.t)
                else:  # Sum with exponentials
                    A2, F2 = self.get_ampfreq(e)
                    for key, value in np.ndenumerate(F2):
                        if not value in FV:
                            FV.append(F2[key[0]])
                            AV.append(A2[key[0]])
                        elif value in FV:
                            i = int(np.argwhere(np.array(FV) == value))
                            AV[i] += A2[key[0]]
                        else:
                            raise('value neither in or not in list')
            if len(FV) == 0:  # if there is no exponential in the expression
                AV = [1]
                FV = [0]
            F += [FV[i] + freq_shift for i in range(len(FV))]
            A += [AV[i]*prefactor for i in range(len(AV))]
        elif type(expr) == sp.core.add.Add:  # expr is first a sum
            for e in expr.args:
                if e.has(sp.exp):  # the term has an exponential
                    amplitude, freq = (e.subs(sp.exp, se)
                                       ).as_coeff_exponent(se)
                    freq = freq/(sp.I*self.t)
                    if freq != 0:
                        if freq in F:
                            i = int(np.argwhere(np.array(F) == freq))
                            A[i] += amplitude
                        elif freq not in F:
                            F.append(freq)
                            A.append(amplitude)
                    else:  # nested sum a*b*(C+d*(U e^() + V )) or constant
                        A2, F2 = self.get_ampfreq(e)
                        for key, value in np.ndenumerate(F2):
                            if not value in F:
                                F.append(F2[key[0]])
                                A.append(A2[key[0]])
                            elif value in F:
                                i = int(np.argwhere(np.array(F) == value))
                                A[i] += A2[key[0]]
                            else:
                                raise('value neither in or not in list')
                else:                               # Constants
                    if 0 in F:
                        i = int(np.argwhere(np.array(F) == 0))
                        A[i] += e
                    else:
                        A.append(e)
                        F.append(0)
        else:  # expr is only an exponential/a constant
            if expr.has(sp.exp):
                amplitude, freq = (expr.subs(sp.exp, se)).as_coeff_exponent(se)
                A = [amplitude]
                F = [freq]
            else:
                A = [expr]
                F = [0]
        return A, F

    def degree(self, expr:sp.Expr, x:sp.Expr):
        """
        Returns the degree of expr wrt to x
        :param expr: expression
        :param x: variable
        :return: degree
        """
        n=1
        while expr.has(x**n):
            n+=1
        return n-1
    
    def moyal_product_legacy(self, o1, o2, trunc=10):
        """
        Returns the Moyal  star product of 2 operators.
        o1 should be a polynomial in a_i
        or o2 should be a polynomial in a_i^dagger
        TODO: the sp.degree function is very cotsly do without it 
        Store the derivative computed at n-1 so to speed up the computation at n.
        :param o1: operator 1
        :param o2: operator 2
        :param trunc: truncation order when the operators are not polynomials
        :return: Moyal star product
        """
        assert np.sum([o1.is_polynomial(self.a[i]) or o2.is_polynomial(
            self.ad[i]) for i in range(len(self.a))]) == len(self.a), 'Bad format in moyal'
        term1 = 0*sp.I
        print('moyal legacy used')
        if o1 == 0 or o2 == 0:
            return 0*sp.I
        # k=np.zeros(len(self.a))
        def my_degree(op, ladder):
            try:
                return int(sp.degree(op, ladder))# Very long
            except:
                #print('degree not found')
                return trunc
        kmax = np.array([np.min((my_degree(o1, self.a[i]), my_degree(
            o2, self.ad[i]))) for i in range(len(self.a))])
        K_array = [tuple([(i//(int(np.prod([kmax[k]+1 for k in range(j)])))) % (kmax[j]+1)
                          for j in range(len(kmax))]) for i in range(np.prod(kmax+1))]
        for key, value in enumerate(K_array):
            list_k = list(value)
            intermediate_left = o1
            intermediate_right = o2
            for i in range(len(self.a)):
                intermediate_left = sp.Derivative(
                    intermediate_left, self.a[i], list_k[i]).doit()
                intermediate_right = sp.Derivative(
                    intermediate_right, self.ad[i], list_k[i]).doit()
            term1 += (self.hbar**np.sum(list_k)/(np.prod([sp.factorial(list_k[i]) for i in range(
                len(list_k))])) * self.formating_rotating(intermediate_left * intermediate_right))
        return term1

    def moyal_product(self, o1, o2, trunc=1):
        """
        Is missing formating
        """
        Do1 = [[o1]]
        Do2 = [[o2]]
        k=0
        while k<trunc:
            k+=1
            Tempo1 = [(Do1[-1][i].diff(self.a[j])) for i in range(len(Do1[-1])) for j in range(self.n_osc)]
            Tempo2 = [(Do2[-1][i].diff(self.ad[j])) for i in range(len(Do2[-1])) for j in range(self.n_osc)]
            Multemps = [(Tempo1[i]*Tempo2[i]) for i in range(len(Tempo1))]# had a delete_constants here, is wrong
            if sum(Multemps)==0:
                break
            ind_zero = np.argwhere([Multemps[i]==0 for i in range(len(Multemps))])
            Do1.append([Tempo1[i] for i in range(len(Tempo1)) if i not in ind_zero])
            Do2.append([Tempo2[i] for i in range(len(Tempo2)) if i not in ind_zero])
        mP = sum([self.hbar**k/(sp.factorial(k)) * (sum([Do1[k][i]*Do2[k][i] for i in range(len(Do1[k]))])) for k in range(len(Do1))])
        return (mP)

    def hussimi_bracket(self, o1, o2):
        """
        Returns the commutator of 2 operators using the Moyal star product
        :param o1: operator 1
        :param o2: operator 2
        :return: commutator/ I*hbar
        """
        if o1.removeO() == 0 or o2.removeO() == 0:
            return self.Ht.getO()
        if hasattr(self,'order_param'):#does not work for now
            coeff_dict1 = o1.removeO().as_coefficients_dict(self.order_param)
            coeff_dict2 = o2.removeO().as_coefficients_dict(self.order_param)
            List = [(self.moyal_product(coeff_dict1[term1],coeff_dict2[term2]) \
            -self.moyal_product(coeff_dict2[term2],coeff_dict1[term1]))/(sp.I*sp.Abs(self.hbar)) \
            *term1*term2 for term2 in coeff_dict2.keys() for term1 in coeff_dict1.keys() if not term1*term2 in self.Ht.getO()]
            return sum(List)
        else:
            return (self.moyal_product(o1, o2)-self.moyal_product(o2, o1))/(sp.I * sp.Abs(self.hbar))

    def RWA_1(self):
        """
        Performs first order RWA. 
        Instanciates the effective hamiltonian, the oscillatory part, the generator and the System-bath hamiltonian
        """
        self.Hb, self.Hosc, self.S1 = 0*self.H0, 0*self.H0, 0*self.H0
        self.Hb = self.not_collect(self.Ht, sp.exp)
        self.Hosc = self.Ht - self.Hb
        self.S1 = -(self.Hosc).replace(sp.exp, lambda x: sp.exp(x) /
                                       x.subs(self.t, 1))  # Integration
        self.Hsb_1 = self.hussimi_bracket(self.S1, self.Hsb)
        self.Hsb_1 = self.Hsb_1

    def RWA_2(self):
        """
        Performs second order RWA. 
        Instanciates the effective hamiltonian, the oscillatory part, the generator and the System-bath hamiltonian
        """
        self.Hb2, self.Hosc2, self.S2 = 0*self.H0, 0*self.H0, 0*self.H0
        dS2 = self.hussimi_bracket(self.S1, self.Hb + 0.5*self.Hosc)
        dS2 = dS2

        self.Hb2 = self.not_collect(dS2, sp.exp)
        self.Hosc2 = (dS2 - self.Hb2)
        self.S2 = -(self.Hosc2).replace(sp.exp,
                                        lambda x: sp.exp(x)/x.subs(self.t, 1))
        self.Hsb_2 = self.hussimi_bracket(
            self.S2, self.Hsb)+(1)/2*self.hussimi_bracket(self.S1, self.Hsb_1)
        self.Hsb_2 = self.Hsb_2

    def RWA_3(self):
        """
        Performs third order RWA. TODO: is not working ... but irrelevant for now
        Instanciates the effective hamiltonian, the oscillatory part, the generator and the System-bath hamiltonian
        """
        self.Hb3, self.Hosc3, self.S3 = np.copy(
            self.Ht)*0, np.copy(self.Ht)*0, np.copy(self.Ht)*0
        dS3 = ((1) * self.hussimi_bracket(self.S2, self.Hb + 0.5*self.Hosc)
               + (1)/2 * self.hussimi_bracket(self.S1, -self.Hosc2 + (1)*self.hussimi_bracket(
                   self.S1, self.Hb + 2/3*self.Hosc)))

        self.Hb3 = self.not_collect(dS3, sp.exp)
        self.Hosc3 = (dS3 - self.Hb3)
        self.S3 = -(self.Hosc3).replace(sp.exp,
                                        lambda x: sp.exp(x)/x.subs(self.t, 1))
        #dS3 = self.formating_rotating(dS3)
        #I = np.array(np.argwhere(dS3!=0))
        # for i in I:
        #     key = tuple(i)
        #     self.Hb3[key] = self.not_collect(dS3[key],sp.exp)
        #     if sum(i) !=0:  # constant terms are going to disappear in the commutator
        #         self.Hosc3[key] = dS3[key] - self.Hb3[key]
        #         self.Hosc3[key] = sp.expand(self.Hosc3[key],power_exp=False)
        #         #Same as formating_rotating method, however it is more efficient to do it here
        #         Expr2 = 0*self.t
        #         if type(self.Hosc3[key])==sp.core.add.Add:
        #             for a in self.Hosc3[key].args:
        #                 Expr2 += sp.powsimp(sp.factor_terms(a),force=True)
        #             self.Hosc3[key] = Expr2
        #     self.S3[key] = -self.Hosc3[key].replace(sp.exp,lambda x: sp.exp(x)/x.subs(self.t,1))  # integration
        # Handle System-bath interaction
        self.Hsb_3 = ((1)*self.hussimi_bracket(self.S3, self.Hsb)
                      + (1)/2*self.hussimi_bracket(self.S2, self.Hsb_1) +
                      (-sp.I)/2*self.hussimi_bracket(self.S1, self.Hsb_2)
                      - (1)**2/12*self.hussimi_bracket(self.S1, self.hussimi_bracket(self.S1, self.Hsb_1)))
        #self.Hsb_3 = self.formating_rotating(self.Hsb_3)

    def print_operator(self, o: sp.Expr):
        """
        Prints the operator o in a nice way
        :param o: operator
        """
        display(Math(sp.latex(o)))

    def JRWA(self, order: int):
        """
        Jaya style RWA 
        seems to work nicely tested on katpizonium with hbar=sp.I 
        :param order: order of the RWA
        """
        self.S = [0*self.H0 for i in range(order+1)]
        self.dS = [0*self.H0 for i in range(order+1)]
        # K0 = K0_0 + K0_1 but Kn = Kn_1 +...+ Kn_(n+1)
        self.Kn_k = [[0*self.H0 for k in range(order+2)]
                     for n in range(order)]
        self.Hn_k = [[0*self.H0 for k in range(order+3)]
                        for n in range(order+1)]
        self.Kn = [0*self.H0 for i in range(order)]
        self.Kn_k[0][0] = self.Ht
        self.Hn_k[0][0] = self.Hsb
        for n in range(1, order+1):
            # Find S[n]
            if n == 1:
                self.Kn[n-1] = self.not_collect(self.Ht, sp.exp) + (self.Ht.getO() or 0)
                self.dS[n] = self.formating_rotating(self.Ht - self.Kn[n-1])
                self.S[n] = -(self.dS[n]).replace(sp.exp,
                                                  lambda x: sp.exp(x)/x.subs(self.t, 1)/self.order_param)  # Integration: DIVIDED BY ORDER PARAMETER
            else:
                Hb = self.hussimi_bracket(self.S[n-1], self.Ht)+ (self.Ht.getO() or 0)
                for k in range(2, n+2):  #  2 to n+1
                    for m in range(n):  # 0 to n-1
                        Hb += 1/k * \
                            self.hussimi_bracket(
                                self.S[n-1-m], self.Kn_k[m][k-1]) + (self.Ht.getO() or 0)
                Hb = self.formating_rotating(self.delete_constants(Hb)) + (self.Ht.getO() or 0)
                self.Kn[n-1] = self.not_collect(Hb, sp.exp) + (self.Ht.getO() or 0)
                self.dS[n] = (Hb - self.Kn[n-1])
                self.S[n] = -(self.dS[n]).replace(sp.exp,
                                                  lambda x: sp.exp(x)/x.subs(self.t, 1)/self.order_param)  # DIVIDED BY ORDER PARAMETER
            # Find Kn_k[n]
            for k in range(1, n+2):  # 1 to n+1
                if k == 1:
                    self.Kn_k[n-1][k] += -self.dS[n] + \
                        self.hussimi_bracket(self.S[n-1], self.Ht) + (self.Ht.getO() or 0)
                else:
                    for m in range(n-1):  # 0 to n-2
                        self.Kn_k[n-1][k] += 1/k * \
                            self.hussimi_bracket(
                                self.S[n-1-m], self.Kn_k[m][k-1]) + (self.Ht.getO() or 0)
            # Handle System-bath interaction
            for k in range(1, n+2):  # 1 to n+1
                if k == 1:
                    self.Hn_k[n-1][k] += self.hussimi_bracket(self.S[n-1], self.Hsb)
                else:
                    for m in range(n-1):
                        self.Hn_k[n-1][k] += 1/k * \
                            self.hussimi_bracket(
                                self.S[n-1-m], self.Hn_k[m][k-1])
        n=order+1
        for k in range(1, n+2):  # 1 to n+1
            if k == 1:
                self.Hn_k[n-1][k] += self.hussimi_bracket(self.S[n-1], self.Hsb)
            else:
                for m in range(n-1):
                    self.Hn_k[n-1][k] += 1/k * \
                        self.hussimi_bracket(
                            self.S[n-1-m], self.Hn_k[m][k-1])
            
    def SymRWA(self, order: int):
        """
        Jaya style symbolic RWA
        :param order: order of the RWA
        """
        self.tavg = sp.symbols('t_{avg}', real=True)

        def osc(exp, t=self.t):
            dexp = sp.diff(exp, t)
            if dexp == 0:
                return 0
            else:
                return sp.factor(exp-exp.subs(t, self.tavg))

        self.S = [0*self.H0 for i in range(order+1)]
        self.dS = [0*self.H0 for i in range(order+1)]
        # K0 = K0_0 + K0_1 but Kn = Kn_1 +...+ Kn_(n+1)
        self.Kn_k = [[0*self.H0 for k in range(order+2)]
                     for n in range(order)]
        self.Kn = [0*self.H0 for i in range(order)]
        self.Kn_k[0][0] = self.Ht
        for n in range(1, order+1):
            # Find S[n]
            if n == 1:
                self.Kn[n-1] = self.Ht.subs(self.t, self.tavg)
                self.dS[n] = osc(self.Ht)
                self.S[n] = -sp.integrate(self.dS[n], (self.t, 0, self.t))
            else:
                Hb = self.hussimi_bracket(self.S[n-1], self.Ht)
                for k in range(2, n+2):  #  2 to n+1
                    for m in range(n):  # 0 to n-1
                        Hb += 1/k * \
                            self.hussimi_bracket(
                                self.S[n-1-m], self.Kn_k[m][k-1])
                self.Kn[n-1] = Hb.subs(self.t, self.tavg)
                self.dS[n] = osc(Hb)
                self.S[n] = -sp.integrate(self.dS[n], (self.t, 0, self.t))
            # Find Kn_k[n]
            for k in range(1, n+2):  # 1 to n+1
                if k == 1:
                    self.Kn_k[n-1][k] += -self.dS[n] + \
                        self.hussimi_bracket(self.S[n-1], self.Ht)
                else:
                    for m in range(n-1):  # 0 to n-2
                        self.Kn_k[n-1][k] += 1/k * \
                            self.hussimi_bracket(
                                self.S[n-1-m], self.Kn_k[m][k-1])

    def Sym2RWA(self, H, order: int):
        """
        Jaya style symbolic RWA
        H: hamiltonian decomposed as [[time function, operator],[time function, operator],...]
        order: order of the RWA
        """
        #self.tavg = sp.symbols('t_{avg}', real=True)
        osc = sp.Function('osc')
        def osc2(expr):
            """
            Takes the formal oscillating part of the expression
            :param expr: expression
            :return: formal oscillating part
            """
            expr = expr + 0*sp.I
            if not expr.has(self.t):
                return 0*sp.I
            else:
                sexpr = sp.expand(expr)
                if sexpr.func.__name__ == 'Add':
                    return sum([osc2(e) for e in sexpr.args])
                elif sexpr.func.__name__ == 'Integral':
                    return sp.Integral(osc2(sexpr.args[0]), sexpr.args[1])
                elif sexpr.func.__name__ == 'osc':
                    return sexpr
                elif sexpr.func.__name__ == 'Mul':
                    m = sp.prod([e for e in sexpr.args if not e.has(self.t)])
                    mt = sp.prod([e for e in sexpr.args if e.has(self.t)])
                    return m*osc(mt)
                else:
                    return osc(sexpr)
        aO = 0*sp.I
        self.Htn = [[[aO, aO]] for i in range(order)]
        self.S = [[[aO, aO]] for i in range(order+1)]
        self.dS = [[[aO, aO]] for i in range(order+1)]
        # K0 = K0_0 + K0_1 but Kn = Kn_1 +...+ Kn_(n+1)
        self.Kn_k = [[[[aO, aO]]
                      for k in range(order+2)] for n in range(order)]
        self.Kn = [[[aO, aO]] for i in range(order)]
        self.Kn_k[0][0] = H  # self.Ht
        for n in range(1, order+1):
            # Find S[n]
            if n == 1:
                self.Kn[n -
                        1] = [[sp.Symbol('\overline{'+sp.latex(h[0])+'}'), h[1]] for h in H]
                self.Htn[n-1] = [[h[0], h[1]] for h in H]
                self.dS[n] = [[-osc2(h[0]), h[1]] for h in H]
                self.S[n] = [[-sp.Integral(s[0], self.t), s[1]]
                             for s in self.dS[n] if s[0] != 0]
            else:
                Hb = [[s[0]*h[0], self.hussimi_bracket(s[1], h[1])]
                      for h in H for s in self.S[n-1] if s[0] != 0]
                for k in range(2, n+2):  #  2 to n+1
                    for m in range(n):  # 0 to n-1
                        for h in self.Kn_k[m][k-1]:
                            for s in self.S[n-1-m]:
                                if s[0] != 0 and h[0] != 0 and s[1] != 0 and h[1] != 0:
                                    Hb.append(
                                        [1/k * s[0]*h[0], 1/k*self.hussimi_bracket(s[1], h[1])])
                # print(Hb)
                self.Kn[n -1] = [[sp.Symbol('\overline{'+sp.latex(h[0])+'}'), h[1]] for h in Hb]
                self.Htn[n-1] = [[h[0], h[1]] for h in Hb]
                # Different definition as above !!!
                self.dS[n] = [[-osc2(h[0]), h[1]] for h in Hb]
                self.S[n] = [[-sp.Integral(s[0], self.t), s[1]]
                             for s in self.dS[n] if s[0] != 0]
            # Find Kn_k[n]
            for k in range(1, n+2):  # 1 to n+1
                if k == 1:
                    self.Kn_k[n-1][k] = self.dS[n] + [[s[0]*h[0],
                                                       self.hussimi_bracket(s[1], h[1])] for h in H for s in self.S[n-1]]
                    #self.Kn_k[n-1][k].append([[s[0]*h[0], self.hussimi_bracket(s[1], h[1])]  for h in H for s in self.S[n-1]])
                else:
                    for m in range(n-1):  # 0 to n-2
                        for h in self.Kn_k[m][k-1]:
                            for s in self.S[n-1-m]:
                                if s[0] != 0 and h[0] != 0 and s[1] != 0 and h[1] != 0:
                                    self.Kn_k[n-1][k].append(
                                        [1/k * s[0]*h[0], 1/k*self.hussimi_bracket(s[1], h[1])])
