from typing import Union, List
import numpy as np

unwrapped_wheel_dist = 0.102
def convert_steering_to_wheel_vels(steering: float, gain=1.0, trim=0.0, radius=0.0318, k=27.0, limit=1.0):
    vel, angle = 0.1, steering
    # Distance between the wheels

    baseline = unwrapped_wheel_dist

    # assuming same motor constants k for both motors
    k_r = k
    k_l = k

    # adjusting k by gain and trim
    k_r_inv = (gain + trim) / k_r
    k_l_inv = (gain - trim) / k_l

    omega_r = (vel - 0.5 * angle * baseline) / radius
    omega_l = (vel + 0.5 * angle * baseline) / radius

    # conversion from motor rotation rate to duty cycle
    u_r = omega_r * k_r_inv
    u_l = omega_l * k_l_inv

    # limiting output to limit, which is 1.0 for the duckiebot
    u_r_limited = max(min(u_r, limit), -limit)
    u_l_limited = max(min(u_l, limit), -limit)

    vels = np.array([u_l_limited, u_r_limited])

    return vels[0], vels[1]

def _perturb(self, val: Union[float, np.ndarray, List[float]], scale: float = 0.1) -> np.ndarray:
    """
    Add noise to a value. This is used for domain randomization.
    """
    assert 0 <= scale < 1

    val = np.array(val)

    if not self.domain_rand:
        return val

    if isinstance(val, np.ndarray):
        noise = self.np_random.uniform(low=1 - scale, high=1 + scale, size=val.shape)
        if val.size == 4:
            noise[3] = 1
    else:
        noise = self.np_random.uniform(low=1 - scale, high=1 + scale)

    res = val * noise

    return res