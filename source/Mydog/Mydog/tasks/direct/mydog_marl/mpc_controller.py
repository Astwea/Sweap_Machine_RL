# mpc_controller.py
import casadi as ca
import numpy as np

class DifferentialDriveMPC:
    def __init__(self, horizon=10, dt=0.1):
        self.horizon = horizon
        self.dt = dt
        self._build_model()

    def wrap_angle(self, angle):
        return ca.fmod(angle + ca.pi, 2 * ca.pi) - ca.pi

    def _build_model(self):
        N = self.horizon
        dt = self.dt
        opti = ca.Opti()

        x = opti.variable(N + 1)
        y = opti.variable(N + 1)
        theta = opti.variable(N + 1)
        v = opti.variable(N)
        w = opti.variable(N)

        x_ref = opti.parameter(N + 1)
        y_ref = opti.parameter(N + 1)
        x0 = opti.parameter()
        y0 = opti.parameter()
        theta0 = opti.parameter()

        # 初始条件
        opti.subject_to(x[0] == x0)
        opti.subject_to(y[0] == y0)
        opti.subject_to(theta[0] == theta0)

        # 动力学约束
        for k in range(N):
            opti.subject_to(x[k + 1] == x[k] + v[k] * ca.cos(theta[k]) * dt)
            opti.subject_to(y[k + 1] == y[k] + v[k] * ca.sin(theta[k]) * dt)
            opti.subject_to(theta[k + 1] == theta[k] + w[k] * dt)

        # 控制范围
        opti.subject_to(opti.bounded(-0.5, v, 0.5))
        opti.subject_to(opti.bounded(-1.5, w, 1.5))

        # 目标函数
        cost = 0
        angle_cost = 0
        pos_cost = 0
        vel_cost = 0
        for k in range(N):
            theta_ref = ca.atan2(y_ref[k] - y[k] +1e-6, x_ref[k] - x[k] +1e-6)
            angle_diff = (theta[k] - theta_ref)

            angle_cost += 5.0 * (1 - ca.cos(angle_diff))**2
            vel_cost += 1.0 * ca.fmax(0, -v[k])**2 + 0.01*v[k]**2
            pos_cost += (x[k] - x_ref[k])**2 + (y[k] - y_ref[k])**2

        cost = angle_cost + pos_cost + vel_cost
        opti.minimize(cost)

        # 求解器配置
        opts = {"ipopt.print_level": 0, "print_time": 0}
        opti.solver("ipopt", opts)

        self.opti = opti
        self.angle_cost = angle_cost
        self.pos_cost = pos_cost
        self.vel_cost = vel_cost
        self.vars = {
            "x": x, "y": y, "theta": theta, "v": v, "w": w,
            "x_ref": x_ref, "y_ref": y_ref,
            "x0": x0, "y0": y0, "theta0": theta0,
        }

    def solve(self, state, ref_traj):
        opti = self.opti
        for var in ["x", "y", "theta", "v", "w"]:
            opti.set_initial(self.vars[var], 0)

        x0, y0, theta0 = state
        ref_traj = np.array(ref_traj)
        x_ref = ref_traj[:, 0]
        y_ref = ref_traj[:, 1]

        opti.set_value(self.vars["x0"], x0)
        opti.set_value(self.vars["y0"], y0)
        opti.set_value(self.vars["theta0"], theta0)
        opti.set_value(self.vars["x_ref"], x_ref)
        opti.set_value(self.vars["y_ref"], y_ref)

        try:
            sol = opti.solve()
            print("angle_cost =", float(sol.value(self.angle_cost)))
            print("pos_cost   =", float(sol.value(self.pos_cost)))
            print("vel_cost   =", float(sol.value(self.vel_cost)))
            v0 = float(sol.value(self.vars["v"][0]))
            w0 = float(sol.value(self.vars["w"][0]))
            return np.array([v0, w0], dtype=np.float32)
        except RuntimeError:
            return np.array([0.0, 0.0], dtype=np.float32)
