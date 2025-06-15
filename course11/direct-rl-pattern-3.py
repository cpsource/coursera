class DrivingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.sensor_processor = nn.Conv2d(...)  # Process camera/lidar
        self.decision_network = nn.Linear(...)   # Same pattern as CartPole!
    
    def forward(self, sensor_data):
        features = self.sensor_processor(sensor_data)
        steering_angle = self.decision_network(features)
        return steering_angle

# In production:
sensor_input = get_camera_and_lidar_data()
steering_command = driving_policy(sensor_input)
send_to_steering_system(steering_command)


