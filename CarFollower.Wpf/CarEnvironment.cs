namespace CarFollower.Wpf;

public class CarEnvironment {
    
    public CarEnvironment(float leaderSpeed, float followerSpeed, float distance) {
        LeaderSpeed = leaderSpeed;
        FollowerSpeed = followerSpeed;
        Distance = distance;
    }

    public void Step(float acceleration) {
        float followerSpeed = Math.Clamp(0, FollowerSpeed + acceleration, 150);
        float leaderSpeed = LeaderSpeed;
        float distance = Distance + (LeaderSpeed - FollowerSpeed) * 1f;
        
        LeaderSpeed = leaderSpeed;
        FollowerSpeed = followerSpeed;
        Distance = distance;
    }
    
    public float FollowerSpeed { get; private set; }
    public float Distance { get; private set; }

    public float LeaderSpeed { get; private set; }
}
