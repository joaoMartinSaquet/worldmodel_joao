import numpy as np
import torch
from scipy.integrate import odeint

def generate_circular_trajectory(num_sequences, seq_len, radius=1.0, noise=0.01):
    """
    Generate circular trajectory data
    
    Args:
        num_sequences: Number of trajectories
        seq_len: Length of each sequence
        radius: Radius of the circle
        noise: Amount of noise to add
    
    Returns:
        List of numpy arrays, each of shape (seq_len, 2)
    """
    trajectories = []
    
    for _ in range(num_sequences):
        # Random starting angle
        start_angle = np.random.uniform(0, 2 * np.pi)
        
        # Generate angles for the trajectory
        angles = np.linspace(start_angle, start_angle + 2 * np.pi, seq_len)
        
        # Generate x, y coordinates
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        
        # Add noise
        x += np.random.normal(0, noise, size=x.shape)
        y += np.random.normal(0, noise, size=y.shape)
        
        # Stack into trajectory
        trajectory = np.stack([x, y], axis=1)
        trajectories.append(trajectory)
    
    return trajectories


def generate_constant_velocity_dataset(num_sequences, seq_len, noise=0.01):
    """
    Generate linear trajectories with constant velocity
    
    Args:
        num_sequences: Number of trajectories
        seq_len: Length of each sequence
        noise: Amount of noise to add
    
    Returns:
        List of numpy arrays, each of shape (seq_len, 2)
    """
    trajectories = []
    
    for _ in range(num_sequences):
        # Random starting position
        start_x = np.random.uniform(-1, 1)
        start_y = np.random.uniform(-1, 1)
        
        # Random velocity
        vx = np.random.uniform(-0.1, 0.1)
        vy = np.random.uniform(-0.1, 0.1)
        
        # Generate trajectory
        t = np.arange(seq_len)
        x = start_x + vx * t
        y = start_y + vy * t
        
        # Add noise
        x += np.random.normal(0, noise, size=x.shape)
        y += np.random.normal(0, noise, size=y.shape)
        
        # Stack into trajectory
        trajectory = np.stack([x, y], axis=1)
        trajectories.append(trajectory)
    
    return trajectories


def generate_sine_wave_dataset(num_sequences, seq_len, noise=0.01):
    """
    Generate sine wave trajectories
    
    Args:
        num_sequences: Number of trajectories
        seq_len: Length of each sequence
        noise: Amount of noise to add
    
    Returns:
        List of numpy arrays, each of shape (seq_len, 2)
    """
    trajectories = []
    
    for _ in range(num_sequences):
        # Random frequency and amplitude
        freq = 2
        amp = 1.0
        phase = np.random.uniform(0, 2 * np.pi)
        
        # Generate trajectory
        t = np.linspace(0, 4 * np.pi, seq_len)
        x = t / (4 * np.pi) * 2 - 1  # Normalize to [-1, 1]
        y = amp * np.sin(freq * t + phase)
        
        # Add noise
        x += np.random.normal(0, noise, size=x.shape)
        y += np.random.normal(0, noise, size=y.shape)
        
        # Stack into trajectory
        trajectory = np.stack([x, y], axis=1)
        trajectories.append(trajectory)
    
    return np.array(trajectories)


def generate_figure_eight_dataset(num_sequences, seq_len, noise=0.01):
    """
    Generate figure-eight trajectories
    
    Args:
        num_sequences: Number of trajectories
        seq_len: Length of each sequence
        noise: Amount of noise to add
    
    Returns:
        List of numpy arrays, each of shape (seq_len, 2)
    """
    trajectories = []
    
    for _ in range(num_sequences):
        # Random starting phase
        phase = np.random.uniform(0, 2 * np.pi)
        scale = np.random.uniform(0.7, 1.3)
        
        # Generate trajectory (Lemniscate of Gerono)
        t = np.linspace(phase, phase + 2 * np.pi, seq_len)
        x = scale * np.cos(t)
        y = scale * np.sin(t) * np.cos(t)
        
        # Add noise
        x += np.random.normal(0, noise, size=x.shape)
        y += np.random.normal(0, noise, size=y.shape)
        
        # Stack into trajectory
        trajectory = np.stack([x, y], axis=1)
        trajectories.append(trajectory)
    
    return trajectories


def generate_dataset(num_sequences, seq_len, types, dt=0.01, noise=0.01):
    """
    Generate mixed dataset with different trajectory types
    
    Args:
        num_sequences: Number of trajectories
        seq_len: Length of each sequence
        dt: Time step (for compatibility)
        noise: Amount of noise to add
    
    Returns:
        List of numpy arrays
    """
    trajectories = []
    
    # Mix different trajectory types
    for _ in range(num_sequences):

        
        if type == 'circular':
            traj = generate_circular_trajectory(1, seq_len, noise=noise)[0]
        elif type == 'linear':
            traj = generate_constant_velocity_dataset(1, seq_len, noise=noise)[0]
        elif type == 'sine':
            traj = generate_sine_wave_dataset(1, seq_len, noise=noise)[0]
        else:  # figure8
            traj = generate_figure_eight_dataset(1, seq_len, noise=noise)[0]
        
        trajectories.append(traj)
    
    return trajectories



class DoublePendulum:
    """
    Double pendulum physics simulator
    
    The double pendulum is a classic example of chaotic dynamics.
    Two pendulums are connected end-to-end, creating complex, unpredictable motion.
    """
    
    def __init__(self, m1=1.0, m2=1.0, L1=1.0, L2=1.0, g=9.81):
        """
        Initialize double pendulum parameters
        
        Args:
            m1: Mass of first pendulum bob (kg)
            m2: Mass of second pendulum bob (kg)
            L1: Length of first pendulum (m)
            L2: Length of second pendulum (m)
            g: Gravitational acceleration (m/s^2)
        """
        self.m1 = m1
        self.m2 = m2
        self.L1 = L1
        self.L2 = L2
        self.g = g
    
    def derivatives(self, state, t):
        """
        Compute derivatives for the double pendulum system
        
        State: [theta1, omega1, theta2, omega2]
        where theta is angle and omega is angular velocity
        """
        theta1, omega1, theta2, omega2 = state
        
        # Differences in angles
        delta = theta2 - theta1
        
        # Denominators for the equations of motion
        den1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * np.cos(delta) * np.cos(delta)
        den2 = (self.L2 / self.L1) * den1
        
        # Equations of motion (Lagrangian mechanics)
        dydt = np.zeros_like(state)
        dydt[0] = omega1  # d(theta1)/dt = omega1
        
        dydt[1] = ((self.m2 * self.L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
                    self.m2 * self.g * np.sin(theta2) * np.cos(delta) +
                    self.m2 * self.L2 * omega2 * omega2 * np.sin(delta) -
                    (self.m1 + self.m2) * self.g * np.sin(theta1)) / den1)
        
        dydt[2] = omega2  # d(theta2)/dt = omega2
        
        dydt[3] = ((-self.m2 * self.L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
                    (self.m1 + self.m2) * self.g * np.sin(theta1) * np.cos(delta) -
                    (self.m1 + self.m2) * self.L1 * omega1 * omega1 * np.sin(delta) -
                    (self.m1 + self.m2) * self.g * np.sin(theta2)) / den2)
        
        return dydt
    
    
    def derivatives(self, state, t):
        theta1, omega1, theta2, omega2 = state

        # Differences in angles
        delta = theta2 - theta1

        # Denominators for the equations of motion
        den1 = (self.m1 + self.m2) * self.L1 - self.m2 * self.L1 * np.cos(delta) * np.cos(delta)
        den2 = (self.L2 / self.L1) * den1

        # Equations of motion (Lagrangian mechanics)
        dydt = np.zeros_like(state)
        dydt[0] = omega1  # d(theta1)/dt = omega1

        # Original equation for d(omega1)/dt
        dydt[1] = ((self.m2 * self.L1 * omega1 * omega1 * np.sin(delta) * np.cos(delta) +
                    self.m2 * self.g * np.sin(theta2) * np.cos(delta) +
                    self.m2 * self.L2 * omega2 * omega2 * np.sin(delta) -
                    (self.m1 + self.m2) * self.g * np.sin(theta1)) / den1) - 1.* omega1

        dydt[2] = omega2  # d(theta2)/dt = omega2

        # Original equation for d(omega2)/dt
        dydt[3] = ((-self.m2 * self.L2 * omega2 * omega2 * np.sin(delta) * np.cos(delta) +
                    (self.m1 + self.m2) * self.g * np.sin(theta1) * np.cos(delta) -
                    (self.m1 + self.m2) * self.L1 * omega1 * omega1 * np.sin(delta) -
                    (self.m1 + self.m2) * self.g * np.sin(theta2)) / den2) - 1. * omega2

        return dydt


    def to_cartesian(self, theta1, theta2):
        """
        Convert angles to Cartesian coordinates
        
        Returns:
            x1, y1: Position of first bob
            x2, y2: Position of second bob
        """
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        
        return x1, y1, x2, y2
    
    def simulate(self, initial_state, t_span, dt=0.01):
        """
        Simulate the double pendulum
        
        Args:
            initial_state: [theta1_0, omega1_0, theta2_0, omega2_0]
            t_span: Total simulation time
            dt: Time step
        
        Returns:
            Dictionary with time, angles, and positions
        """
        t = np.arange(0, t_span, dt)
        
        # Solve the differential equations
        states = odeint(self.derivatives, initial_state, t)
        
        theta1 = states[:, 0]
        omega1 = states[:, 1]
        theta2 = states[:, 2]
        omega2 = states[:, 3]
        
        # Convert to Cartesian coordinates
        x1, y1, x2, y2 = self.to_cartesian(theta1, theta2)
        
        return {
            't': t,
            'theta1': theta1,
            'omega1': omega1,
            'theta2': theta2,
            'omega2': omega2,
            'x1': x1,
            'y1': y1,
            'x2': x2,
            'y2': y2
        }


def generate_double_pendulum_dataset(num_sequences, seq_len, dt=0.01, 
                                     output_type='tip_position',
                                     chaos_level='medium',
                                     noise=0.0):
    """
    Generate dataset of double pendulum trajectories
    
    Args:
        num_sequences: Number of trajectories to generate
        seq_len: Length of each sequence
        dt: Time step
        output_type: What to output as observations
            - 'tip_position': (x2, y2) - position of second bob only
            - 'both_positions': (x1, y1, x2, y2) - both bob positions
            - 'angles': (theta1, theta2) - joint angles
            - 'full_state': (theta1, omega1, theta2, omega2) - complete state
        chaos_level: Initial condition spread
            - 'low': Small perturbations, more periodic
            - 'medium': Moderate chaos
            - 'high': Maximum chaos, very sensitive
        noise: Gaussian noise level to add to observations
    
    Returns:
        List of numpy arrays containing trajectories
    """
    trajectories = []
    
    # Set initial condition ranges based on chaos level
    if chaos_level == 'low':
        theta_range = np.pi / 6  # ±30 degrees
        omega_range = 0.5
    elif chaos_level == 'medium':
        theta_range = np.pi / 3  # ±60 degrees
        omega_range = 1.0
    else:  # high
        theta_range = np.pi  # ±180 degrees
        omega_range = 2.0
    
    for _ in range(num_sequences):
        # Random initial conditions
        theta1_0 = np.random.uniform(-theta_range, theta_range)
        theta2_0 = np.random.uniform(-theta_range, theta_range)
        omega1_0 = np.random.uniform(-omega_range, omega_range)
        omega2_0 = np.random.uniform(-omega_range, omega_range)
        
        initial_state = [theta1_0, omega1_0, theta2_0, omega2_0]
        
        # Random pendulum parameters for variation
        m1 = np.random.uniform(0.8, 1.2)
        m2 = np.random.uniform(0.8, 1.2)
        L1 = np.random.uniform(0.8, 1.2)
        L2 = np.random.uniform(0.8, 1.2)
        
        pendulum = DoublePendulum(m1=m1, m2=m2, L1=L1, L2=L2)
        
        # Simulate
        t_span = seq_len * dt
        result = pendulum.simulate(initial_state, t_span, dt)
        
        # Extract observations based on output_type
        if output_type == 'tip_position':
            obs = np.stack([result['x2'], result['y2']], axis=1)
        elif output_type == 'both_positions':
            obs = np.stack([result['x1'], result['y1'], 
                           result['x2'], result['y2']], axis=1)
        elif output_type == 'angles':
            obs = np.stack([result['theta1'], result['theta2']], axis=1)
        else:  # full_state
            obs = np.stack([result['theta1'], result['omega1'],
                           result['theta2'], result['omega2']], axis=1)
        
        # Take only seq_len samples
        obs = obs[:seq_len]
        
        # Add noise if specified
        if noise > 0:
            obs += np.random.normal(0, noise, size=obs.shape)
        
        trajectories.append(obs)
    
    return trajectories


def generate_simple_pendulum_dataset(num_sequences, seq_len, dt=0.01,
                                     damping=0.0, noise=0.0):
    """
    Generate simple (single) pendulum dataset for comparison
    
    Args:
        num_sequences: Number of trajectories
        seq_len: Length of each sequence
        dt: Time step
        damping: Damping coefficient (0 = no damping)
        noise: Observation noise level
    
    Returns:
        List of trajectories (x, y) positions
    """
    trajectories = []
    g = 9.81
    L = 1.0
    
    for _ in range(num_sequences):
        # Random initial conditions
        theta0 = np.random.uniform(-np.pi/2, np.pi/2)
        omega0 = np.random.uniform(-2, 2)
        
        t = np.linspace(0, seq_len * dt, seq_len)
        
        # Simple pendulum differential equation
        def derivatives(state, t):
            theta, omega = state
            dtheta = omega
            domega = -(g/L) * np.sin(theta) - damping * omega
            return [dtheta, domega]
        
        states = odeint(derivatives, [theta0, omega0], t)
        theta = states[:, 0]
        
        # Convert to Cartesian
        x = L * np.sin(theta)
        y = -L * np.cos(theta)
        
        obs = np.stack([x, y], axis=1)
        
        if noise > 0:
            obs += np.random.normal(0, noise, size=obs.shape)
        
        trajectories.append(obs)
    
    return trajectories


def generate_coupled_oscillators_dataset(num_sequences, seq_len, dt=0.01,
                                         coupling_strength=0.5, noise=0.0):
    """
    Generate coupled harmonic oscillators dataset
    
    Two masses connected by springs - exhibits interesting phase relationships
    
    Args:
        num_sequences: Number of trajectories
        seq_len: Length of each sequence
        dt: Time step
        coupling_strength: Strength of coupling between oscillators
        noise: Observation noise
    
    Returns:
        List of trajectories (x1, x2) positions
    """
    trajectories = []
    
    for _ in range(num_sequences):
        # Random initial conditions
        x1_0 = np.random.uniform(-1, 1)
        v1_0 = np.random.uniform(-1, 1)
        x2_0 = np.random.uniform(-1, 1)
        v2_0 = np.random.uniform(-1, 1)
        
        t = np.linspace(0, seq_len * dt, seq_len)
        
        # Coupled oscillator equations
        def derivatives(state, t):
            x1, v1, x2, v2 = state
            
            # Natural frequencies
            omega1 = np.random.uniform(1, 3)
            omega2 = np.random.uniform(1, 3)
            
            dx1 = v1
            dv1 = -omega1**2 * x1 + coupling_strength * (x2 - x1)
            dx2 = v2
            dv2 = -omega2**2 * x2 + coupling_strength * (x1 - x2)
            
            return [dx1, dv1, dx2, dv2]
        
        states = odeint(derivatives, [x1_0, v1_0, x2_0, v2_0], t)
        
        x1 = states[:, 0]
        x2 = states[:, 2]
        
        obs = np.stack([x1, x2], axis=1)
        
        if noise > 0:
            obs += np.random.normal(0, noise, size=obs.shape)
        
        trajectories.append(obs)
    
    return trajectories




if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # Test the generators
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    # Circular
    circular = generate_circular_trajectory(5, 100)
    for traj in circular:
        axes[0, 0].plot(traj[:, 0], traj[:, 1], alpha=0.6)
    axes[0, 0].set_title('Circular Trajectories')
    axes[0, 0].set_aspect('equal')
    
    # Linear
    linear = generate_constant_velocity_dataset(5, 100)
    for traj in linear:
        axes[0, 1].plot(traj[:, 0], traj[:, 1], alpha=0.6)
    axes[0, 1].set_title('Linear Trajectories')
    axes[0, 1].set_aspect('equal')
    
    # Sine
    sine = generate_sine_wave_dataset(5, 100)
    for traj in sine:
        axes[1, 0].plot(traj[:, 0], traj[:, 1], alpha=0.6)
    axes[1, 0].set_title('Sine Wave Trajectories')
    axes[1, 0].set_aspect('equal')
    
    # Figure-eight
    figure8 = generate_figure_eight_dataset(5, 100)
    for traj in figure8:
        axes[1, 1].plot(traj[:, 0], traj[:, 1], alpha=0.6)
    axes[1, 1].set_title('Figure-Eight Trajectories')
    axes[1, 1].set_aspect('equal')
    
    plt.tight_layout()
    # plt.savefig('/home/claude/trajectory_examples.png', dpi=150, bbox_inches='tight')
    plt.show()    
    print("Generated example trajectories saved to trajectory_examples.png")
