using System;
using ApiBackend.Api.Models;

namespace ApiBackend.Api.RateLimiting
{
    [AttributeUsage(AttributeTargets.Method | AttributeTargets.Class, AllowMultiple = true)]
    public class RateLimitPerRoleAttribute : Attribute
    {
        public int LimitRequests { get; }
        public int WindowSeconds { get; }
        public Roles Role { get; }

        public RateLimitPerRoleAttribute(int limitRequests, int windowSeconds, Roles role)
        {
            LimitRequests = limitRequests;
            WindowSeconds = windowSeconds;
            Role = role;
        }
    }

}