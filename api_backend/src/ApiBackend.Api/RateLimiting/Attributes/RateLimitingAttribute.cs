using System;


namespace ApiBackend.Api.RateLimiting
{
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.All)]
public class RateLimitAttribute : Attribute {
  public int LimitRequests { get; }
  public int WindowSeconds { get; }
 
  public RateLimitAttribute(int limitRequests, int windowSeconds){
    LimitRequests = limitRequests;
    WindowSeconds = windowSeconds;
  }
}
}
