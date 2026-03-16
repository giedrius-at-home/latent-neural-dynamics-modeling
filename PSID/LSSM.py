"""
Copyright (c) 2020 University of Southern California
See full notice in LICENSE.md
Omid G. Sani and Maryam M. Shanechi
Shanechi Lab, University of Southern California

An LSSM object for keeping parameters, filtering, etc
"""

import warnings

import numpy as np
from scipy import linalg


def dict_get_either(d, fieldNames, defaultVal=None):
    for f in fieldNames:
        if f in d:
            return d[f]
    return defaultVal


def genRandomGaussianNoise(N, Q, m=None):
    Q2 = np.atleast_2d(Q)
    dim = Q2.shape[0]
    if m is None:
        m = np.zeros((dim, 1))

    D, V = linalg.eig(Q2)
    if np.any(D < 0):
        raise ("Cov matrix is not PSD!")
    QShaping = np.real(np.matmul(V, np.sqrt(np.diag(D))))
    w = np.matmul(np.random.randn(N, dim), QShaping.T)
    return w, QShaping


class LSSM:
    def __init__(self, params, output_dim=None, state_dim=None, input_dim=None):
        self.output_dim = output_dim
        self.state_dim = state_dim
        self.input_dim = input_dim
        # Modified by Giedrius
        self.A_powers_cache = []  # Cache for precomputed A matrix powers
        self.max_m_precomputed = 0  # Maximum m for which A powers are computed
        self.setParams(params)

    def setParams(self, params={}):
        A = dict_get_either(params, ["A", "a"])
        A = np.atleast_2d(A)

        C = dict_get_either(params, ["C", "c"])
        C = np.atleast_2d(C)

        self.A = A
        self.state_dim = self.A.shape[0]
        if C.shape[1] != self.state_dim and C.shape[0] == self.state_dim:
            C = C.T
        self.C = C
        self.output_dim = self.C.shape[0]

        B = dict_get_either(params, ["B", "b"], None)
        D = dict_get_either(params, ["D", "d", "Dy", "dy"], None)
        if isinstance(B, float) or (isinstance(B, np.ndarray) and B.size > 0):
            B = np.atleast_2d(B)
            if B.shape[0] != self.state_dim:
                B = B.T
            self.input_dim = B.shape[1]
        elif isinstance(D, float) or (isinstance(D, np.ndarray) and D.size > 0):
            D = np.atleast_2d(D)
            if D.shape[0] != self.output_dim:
                D = D.T
            self.input_dim = D.shape[1]
        else:
            self.input_dim = 0
        if B is None or B.size == 0:
            B = np.zeros((self.state_dim, self.input_dim))
        B = np.atleast_2d(B)
        if (
            B.size > 0
            and B.shape[0] != self.state_dim
            and B.shape[1] == self.output_dim
        ):
            B = B.T
        self.B = B
        if D is None or D.size == 0:
            D = np.zeros((self.output_dim, self.input_dim))
        D = np.atleast_2d(D)
        if (
            D.size > 0
            and D.shape[0] != self.output_dim
            and D.shape[1] == self.output_dim
        ):
            D = D.T
        self.D = D

        if "q" in params or "Q" in params:  # Stochastic form with QRS provided
            Q = dict_get_either(params, ["Q", "q"], None)
            R = dict_get_either(params, ["R", "r"], None)
            S = dict_get_either(params, ["S", "s"], None)
            Q = np.atleast_2d(Q)
            R = np.atleast_2d(R)

            self.Q = Q
            self.R = R
            if S is None or S.size == 0:
                S = np.zeros((self.state_dim, self.output_dim))
            S = np.atleast_2d(S)
            if S.shape[0] != self.state_dim:
                S = S.T
            self.S = S
        elif "k" in params or "K" in params:
            self.Q = None
            self.R = None
            self.S = None
            self.K = np.atleast_2d(dict_get_either(params, ["K", "k"], None))
            self.innovCov = np.atleast_2d(dict_get_either(params, ["innovCov"], None))

        self.update_secondary_params()

        for f, v in params.items():  # Add any remaining params (e.g. Cz)
            if f in set(["Cz", "Dz"]) or (
                not hasattr(self, f)
                and not hasattr(self, f.upper())
                and f not in set(["sig", "L0", "P"])
            ):
                setattr(self, f, v)

        if hasattr(self, "Cz") and self.Cz is not None:
            Cz = np.atleast_2d(self.Cz)
            if Cz.shape[1] != self.state_dim and Cz.shape[0] == self.state_dim:
                Cz = Cz.T
                self.Cz = Cz

    def changeParams(self, params={}):
        curParams = self.getListOfParams()
        for f, v in curParams.items():
            if f not in params:
                params[f] = v
        self.setParams(params)

    def getListOfParams(self):
        params = {}
        for field in dir(self):
            val = self.__getattribute__(field)
            if not field.startswith("__") and isinstance(
                val, (np.ndarray, list, tuple, type(self))
            ):
                params[field] = val
        return params

    def update_secondary_params(self):
        if self.Q is not None and self.state_dim > 0:  # Given QRS
            try:
                A_Eigs = linalg.eig(self.A)[0]
            except Exception as e:
                print("Error in eig ({})... Trying again!".format(e))
                A_Eigs = linalg.eig(self.A)[0]  # Try again!
            isStable = np.max(np.abs(A_Eigs)) < 1
            if isStable:
                self.XCov = linalg.solve_discrete_lyapunov(self.A, self.Q)
                self.G = self.A @ self.XCov @ self.C.T + self.S
                self.YCov = self.C @ self.XCov @ self.C.T + self.R
                self.YCov = (self.YCov + self.YCov.T) / 2
            else:
                self.XCov = np.eye(self.state_dim)
                self.XCov[:] = np.nan
                self.YCov = np.eye(self.output_dim)
                self.YCov[:] = np.nan

            try:
                # Modified by Giedrius, set the explicit params
                self.Pp = linalg.solve_discrete_are(
                    a=self.A.T, b=self.C.T, q=self.Q, r=self.R, e=None, s=self.S
                )  # Solves Katayama eq. 5.42a
                self.innovCov = self.C @ self.Pp @ self.C.T + self.R
                innovCovInv = np.linalg.pinv(self.innovCov)
                self.K = (self.A @ self.Pp @ self.C.T + self.S) @ innovCovInv
                self.Kf = self.Pp @ self.C.T @ innovCovInv
                self.Pf = self.Pp - self.Kf @ self.C @ self.Pp  # Modified by Giedrius
                self.Cs = (
                    self.Pf @ self.A.T @ np.linalg.pinv(self.Pp)
                )  # Modified by Giedrius
                self.Kv = self.S @ innovCovInv
                self.A_KC = self.A - self.K @ self.C
            except Exception as err:
                print("Could not solve DARE: {}".format(err))
                self.Pp = np.empty(self.A.shape)
                self.Pp[:] = np.nan
                self.K = np.empty((self.A.shape[0], self.R.shape[0]))
                self.K[:] = np.nan
                self.Kf = np.array(self.K)
                self.Kv = np.array(self.K)
                self.innovCov = np.empty(self.R.shape)
                self.innovCov[:] = np.nan
                self.A_KC = np.empty(self.A.shape)
                self.A_KC[:] = np.nan

            self.P2 = (
                self.XCov - self.Pp
            )  # (should give the solvric solution) Proof: Katayama Theorem 5.3 and A.3 in pvo book
        elif hasattr(self, "K") and self.K is not None:  # Given K
            self.XCov = None
            if not hasattr(self, "G"):
                self.G = None
            if not hasattr(self, "YCov"):
                self.YCov = None

            self.Pp = None
            self.Kf = None
            self.Kv = None
            self.A_KC = self.A - self.K @ self.C
            if not hasattr(self, "P2"):
                self.P2 = None
        elif self.R is not None:
            self.YCov = self.R

    def isStable(self):
        return np.all(np.abs(self.eigenvalues) < 1)

    def generateObservationFromStates(
        self, X, u=None, param_names=["C", "D"], prep_model_param="YPrepModel"
    ):
        """Can generate Y or Z observation time series given the latent state time series X and optional external input u

        Args:
            X (numpy array): Dimensions are time x dimesions.
            param_names (list, optional): The name of the read-out parameter. Defaults to ['C'].
            prep_model_param (str, optional): The name of the preprocessing model parameter.
                        Defaults to 'YPrepModel'.

        Returns:
            numpy array: The observation time series.
                If param_names=['C'] and prep_model_param='YPrepModel', will
                    produce Y = C * X, and then applies the inverse of the
                    Y preprocessing model.
                If param_names=['Cz'] and prep_model_param='ZPrepModel', will
                    produce Y = Cz * X, and then applies the inverse of the
                    Z preprocessing model.
        """
        Y = None
        if hasattr(self, param_names[0]):
            C = getattr(self, param_names[0])
        else:
            C = None
        if len(param_names) > 1 and hasattr(self, param_names[1]):
            D = getattr(self, param_names[1])
        else:
            D = None

        if C is not None and C.size > 0 or D is not None and D.size > 0:
            ny = C.shape[0] if C is not None and self.C.size > 0 else D.shape[0]
            N = X.shape[0]
            Y = np.zeros((N, ny))
            if C is not None and C.size > 0:
                Y += (C @ X.T).T
            if D is not None and D.size > 0 and u is not None:
                if hasattr(self, "UPrepModel") and self.UPrepModel is not None:
                    u = self.UPrepModel.apply(
                        u, time_first=True
                    )  # Apply any mean removal/zscoring
                Y += (D @ u.T).T

        if prep_model_param is not None and hasattr(self, prep_model_param):
            prep_model_param_obj = getattr(self, prep_model_param)
            if prep_model_param_obj is not None:
                Y = prep_model_param_obj.apply_inverse(
                    Y
                )  # Apply inverse of any mean-removal/zscoring

        return Y

    def generateRealization(
        self, N, x0=None, w0=None, u0=None, u=None, return_wv=False
    ):
        QRS = np.block([[self.Q, self.S], [self.S.T, self.R]])
        wv, self.QRSShaping = genRandomGaussianNoise(N, QRS)
        w = wv[:, : self.state_dim]
        v = wv[:, self.state_dim :]
        if x0 is None:
            x0 = np.zeros((self.state_dim, 1))
        if w0 is None:
            w0 = np.zeros((self.state_dim, 1))
        if self.input_dim > 0 and u0 is None:
            u0 = np.zeros((self.input_dim, 1))
        X = np.empty((N, self.state_dim))
        Y = np.empty((N, self.output_dim))
        for i in range(N):
            if i == 0:
                Xt_1 = x0
                Wt_1 = w0
                if self.input_dim > 0 and u is not None:
                    Ut_1 = u0
            else:
                Xt_1 = X[i - 1, :].T
                Wt_1 = w[i - 1, :].T
                if self.input_dim > 0 and u is not None:
                    Ut_1 = u[i - 1, :].T
            X[i, :] = (self.A @ Xt_1 + Wt_1).T
            # Y[i, :] = (self.C @ X[i, :].T + v[i, :].T).T # Will make Y later
            if u is not None:
                X[i, :] += np.squeeze((self.B @ Ut_1).T)
                # Y[i, :] += np.squeeze((self.D @ u[i, :]).T) # Will make Y later
        Y = v
        CxDu = self.generateObservationFromStates(
            X, u=u, param_names=["C", "D"], prep_model_param="YPrepModel"
        )
        if CxDu is not None:
            Y += CxDu
        out = Y, X
        if return_wv:
            out += (wv,)
        return out

    def kalman(
        self, Y, U=None, x0=None, P0=None, steady_state=True, return_state_cov=False
    ):
        """Applies the Kalman filter associated with the LSSM to some observation time-series

        Args:
            Y (np.ndarray): observation time series (time first).
            U (np.ndarray, optional): input time series (time first). Defaults to None.
            x0 (np.ndarray, optional): Initial Kalman state. Defaults to None.
            P0 (np.ndarray, optional): Initial Kalman state estimation error covariance. Defaults to None.
            steady_state (bool, optional): If True, will use steady state Kalman gain, which is much faster. Defaults to True.
            return_state_cov (bool, optional): If true, will return state error covariances. Defaults to False.

        Returns:
            allXp (np.ndarray): one-step ahead predicted states (t|t-1)
            allYp (np.ndarray): one-step ahead predicted observations (t|t-1)
            allXf (np.ndarray): filtered states (t|t)
            allPp (np.ndarray): error cov for one-step ahead predicted states (t|t-1)
            allPf (np.ndarray): error cov for filtered states (t|t)
        """
        if self.state_dim == 0:
            allXp = np.zeros((Y.shape[0], self.state_dim))
            allXf = allXp
            allYp = np.zeros((Y.shape[0], self.output_dim))
            return allXp, allYp, allXf
        if np.any(np.isnan(self.K)) and steady_state:
            steady_state = False
            warnings.warn(
                "Steady state Kalman gain not available. Will perform non-steady-state Kalman."
            )
        N, ny = Y.shape[0], Y.shape[1]
        allXp = np.nan * np.ones((N, self.state_dim))  # X(i|i-1)
        allXf = np.nan * np.ones((N, self.state_dim))  # X(i|i)
        if return_state_cov:
            allPp = np.zeros((N, self.state_dim, self.state_dim))  # P(i|i-1)
            allPf = np.zeros((N, self.state_dim, self.state_dim))  # P(i|i)
        if x0 is None:
            x0 = np.zeros((self.state_dim, 1))
        if P0 is None:
            P0 = np.eye(self.state_dim)
        Xp = x0
        Pp = P0
        for i in range(N):
            allXp[i, :] = np.transpose(Xp)  # X(i|i-1)
            thisY = Y[i, :][np.newaxis, :]
            if hasattr(self, "YPrepModel") and self.YPrepModel is not None:
                thisY = self.YPrepModel.apply(
                    thisY, time_first=True
                )  # Apply any mean removal/zscoring
            zi = thisY.T - self.C @ Xp  # Innovation Z(i)
            if U is not None:
                ui = U[i, :][:, np.newaxis]
                if hasattr(self, "UPrepModel") and self.UPrepModel is not None:
                    ui = self.UPrepModel.apply(
                        ui, time_first=False
                    )  # Apply any mean removal/zscoring
                if self.D.size > 0:
                    zi -= self.D @ ui

            if steady_state:
                Kf = self.Kf
                K = self.K
            else:
                ziCov = self.C @ Pp @ self.C.T + self.R
                Kf = np.linalg.lstsq(ziCov.T, (Pp @ self.C.T).T, rcond=None)[
                    0
                ].T  # Kf(i)

                if self.S.size > 0:
                    Kw = np.linalg.lstsq(ziCov.T, self.S.T, rcond=None)[0].T  # Kw(i)
                    K = self.A @ Kf + Kw  # K(i)
                else:
                    K = self.A @ Kf  # K(i)

                P = Pp - Kf @ self.C @ Pp  # P(i|i)

                if return_state_cov:
                    allPp[i, :, :] = Pp  # P(i|i-1)
                    allPf[i, :, :] = P  # P(i|i)

            if Kf is not None:  # Otherwise cannot do filtering
                X = Xp + Kf @ zi  # X(i|i)
                allXf[i, :] = np.transpose(X)

            newXp = self.A @ Xp
            newXp += K @ zi
            if U is not None and self.B.size > 0:
                newXp += self.B @ ui

            Xp = newXp
            if not steady_state:
                Pp = self.A @ Pp @ self.A.T + self.Q - K @ ziCov @ K.T

        allYp = self.generateObservationFromStates(
            allXp, u=U, param_names=["C", "D"], prep_model_param="YPrepModel"
        )

        if not return_state_cov:
            return allXp, allYp, allXf
        else:
            return allXp, allYp, allXf, allPp, allPf

    def smooth(self, Y, U=None, steady_state=True, **kwargs):  # Modified by Giedrius
        """Applies the RTS smoother associated with the LSSM # Modified by Giedrius"""
        if self.state_dim == 0:  # Modified by Giedrius
            allXp = np.zeros((Y.shape[0], self.state_dim))  # Modified by Giedrius
            allYp = np.zeros((Y.shape[0], self.output_dim))  # Modified by Giedrius
            return None, allYp, allXp  # Modified by Giedrius

        # Forward pass # Modified by Giedrius
        if not steady_state:  # Modified by Giedrius
            allXp, allYp, allXf, allPp, allPf = self.kalman(
                Y, U=U, return_state_cov=True, steady_state=False, **kwargs
            )  # Modified by Giedrius
        else:  # Modified by Giedrius
            allXp, allYp, allXf = self.kalman(
                Y, U=U, steady_state=True, **kwargs
            )  # Modified by Giedrius

        N = allXf.shape[0]  # Modified by Giedrius
        nx = self.state_dim  # Modified by Giedrius
        allXs = np.zeros((N, nx))  # Modified by Giedrius
        allXs[-1, :] = allXf[-1, :]  # Modified by Giedrius

        # Backward pass (RTS) # Modified by Giedrius
        for t in range(N - 2, -1, -1):  # Modified by Giedrius
            if not steady_state:  # Modified by Giedrius
                Ct = (
                    allPf[t] @ self.A.T @ np.linalg.pinv(allPp[t + 1])
                )  # Modified by Giedrius
            else:  # Modified by Giedrius
                Ct = self.Cs  # Modified by Giedrius

            # x_s[t] = x_f[t] + Ct @ (x_s[t+1] - x_p[t+1]) # Modified by Giedrius
            allXs[t, :] = allXf[t, :] + (
                Ct @ (allXs[t + 1, :] - allXp[t + 1, :])
            )  # Modified by Giedrius

        allYs = self.generateObservationFromStates(
            allXs, u=U, param_names=["C", "D"], prep_model_param="YPrepModel"
        )  # Modified by Giedrius
        if (hasattr(self, "Cz") and self.Cz is not None) or (
            hasattr(self, "Dz") and self.Dz is not None
        ):  # Modified by Giedrius
            allZs = self.generateObservationFromStates(
                allXs, u=U, param_names=["Cz", "Dz"], prep_model_param="ZPrepModel"
            )  # Modified by Giedrius
        else:  # Modified by Giedrius
            allZs = None  # Modified by Giedrius

        return allZs, allYs, allXs  # Modified by Giedrius

    def predict(
        self, Y, U=None, useXFilt=False, useSmoothing=False, **kwargs
    ):  # Modified by Giedrius
        if isinstance(Y, (list, tuple)):  # If segments of data are provided as a list
            for trialInd, trialY in enumerate(Y):
                trialOuts = self.predict(
                    trialY,
                    U=U if U is None else U[trialInd],
                    useXFilt=useXFilt,
                    useSmoothing=useSmoothing,  # Modified by Giedrius
                    **kwargs,
                )
                if trialInd == 0:
                    outs = [[o] for oi, o in enumerate(trialOuts)]
                else:
                    outs = [outs[oi] + [o] for oi, o in enumerate(trialOuts)]
            return tuple(outs)
        # If only one data segment is provided
        if useSmoothing:  # Modified by Giedrius
            return self.smooth(Y, U=U, **kwargs)  # Modified by Giedrius

        allXp, allYp, allXf = self.kalman(Y, U=U, **kwargs)[0:3]
        if useXFilt:
            allXp = allXf
        if (hasattr(self, "Cz") and self.Cz is not None) or (
            hasattr(self, "Dz") and self.Dz is not None
        ):
            allZp = self.generateObservationFromStates(
                allXp, u=U, param_names=["Cz", "Dz"], prep_model_param="ZPrepModel"
            )
        else:
            allZp = None

        return allZp, allYp, allXp

    def forecast(self, m, Y_past=None, U_past=None, U_future=None):
        # Modified by Giedrius
        """Forecasts the model m steps into the future using deterministic state propagation.

        The forecast works by:
        1. Get initial state from Kalman filter on past data
        2. Propagate state forward: X_{t+1} = A @ X_t
        3. Compute outputs: Y_t = C @ X_t, Z_t = Cz @ X_t

        Args:
            m (int): The number of steps to forecast.
            Y_past (np.ndarray or list, optional): Past neural data to initialize
                the state. (time x dim) or list of segments.
            U_past (np.ndarray or list, optional): Past input data (for IPSID)
                to initialize the state. (time x dim) or list of segments.
            U_future (np.ndarray or list, optional): Future input data (for IPSID)
                required for forecasting. Must have m steps. (m x dim) or list.

        Returns:
            allZp (np.ndarray or list): m-step forecast of behavior Z.
            allYp (np.ndarray or list): m-step forecast of neural data Y.
            allXp (np.ndarray or list): m-step forecast of latent state X.
        """        

        if isinstance(Y_past, (list, tuple)):
            outs = []
            for i in range(len(Y_past)):
                U_future_seg = U_future[i] if U_future is not None else None
                U_past_seg = U_past[i] if U_past is not None else None

                trial_outs = self.forecast(
                    m,
                    Y_past=Y_past[i],
                    U_past=U_past_seg,
                    U_future=U_future_seg,
                )
                if i == 0:
                    outs = [[o] for o in trial_outs]
                else:
                    outs = [outs[oi] + [o] for oi, o in enumerate(trial_outs)]
            return tuple(outs)

        # Get initial states from Kalman filter
        allXp, _, allXf = self.kalman(Y_past, U=U_past)
        
        # Extract state-space matrices
        A = np.array(self.A)
        
        # Start from filtered state X(t|t) - the best estimate after seeing all past observations
        # This avoids discontinuity at the start of forecast
        Xf_last = allXf[-1:].T  # Last filtered state (column vector) - this is X(t|t)
        # Initialize predicted state for first forecast step: X(t+1|t) = A @ X(t|t)
        Xp = A @ Xf_last
        Cy = np.array(self.C)
        Cz = np.array(self.Cz) if hasattr(self, 'Cz') and self.Cz is not None else None

        # Initialize forecast arrays
        nx = Xp.shape[0]
        ny = Cy.shape[0]
        nz = Cz.shape[0] if Cz is not None else 0

        Xf = np.zeros((m, nx))
        Yf = np.zeros((m, ny))
        Zf = np.zeros((m, nz)) if nz > 0 else None

        # Forecast m steps ahead using deterministic state propagation
        for step in range(m):
            # Store forecasted state
            Xf[step] = Xp.flatten()
            
            # Predict observation
            Yp = Cy @ Xp
            Yf[step] = Yp.flatten()
            
            if Zf is not None:
                Zf[step] = (Cz @ Xp).flatten()

            # Propagate state forward for next step
            Xp = A @ Xp

            # Handle future inputs if provided (for IPSID)
            if U_future is not None and hasattr(self, 'B') and self.B is not None and self.B.size > 0:
                if step < U_future.shape[0]:
                    ui = U_future[step:step+1, :].T
                    if hasattr(self, "UPrepModel") and self.UPrepModel is not None:
                        ui = self.UPrepModel.apply(ui, time_first=False)
                    Xp = Xp + self.B @ ui

        return Zf, Yf, Xf
