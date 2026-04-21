from simple_pid import PID

pid = PID(Kp=1.0, Ki=0.1, Kd=0.05, setpoint=0.1)

def current_control(disruption_prob):
    """Regulates current via PID to avoid disruptions."""
    return pid(disruption_prob)

