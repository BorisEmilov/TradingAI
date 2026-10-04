using ApiBackend.Api.Data;
using ApiBackend.Api.Interfaces;
using ApiBackend.Api.Repositories;
using ApiBackend.Api.Services;
using ApiBackend.Api.RateLimiting;
using Microsoft.AspNetCore.Authentication.JwtBearer;
using Microsoft.EntityFrameworkCore;
using Microsoft.IdentityModel.Tokens;
using System.Text;
using StackExchange.Redis;



var builder = WebApplication.CreateBuilder(args);


// ! *******************************************
// ! ********** BUILDER ******************
// ! *******************************************

builder.Services.AddDbContext<ApplicationDBContext>(options =>
    options.UseSqlServer(builder.Configuration.GetConnectionString("DefaultConnection")));

builder.Services.AddScoped<TokenService>();
builder.Services.AddScoped<IUserRepository, UserRepository>();
builder.Services.AddScoped<IPlanRepository, PlanRepository>();
builder.Services.AddScoped<IMtAccountRepository, MtAccountRepository>();
builder.Services.AddScoped<SendEmailService>();

builder.Services.AddAuthentication(JwtBearerDefaults.AuthenticationScheme)
    .AddJwtBearer(options =>
    {
        options.TokenValidationParameters = new TokenValidationParameters
        {
            ValidateIssuer = true,
            ValidateAudience = true,
            ValidateLifetime = true,
            ValidateIssuerSigningKey = true,
            ValidIssuer = builder.Configuration["Jwt:Issuer"],
            ValidAudience = builder.Configuration["Jwt:Audience"],
            IssuerSigningKey = new SymmetricSecurityKey(
                Encoding.UTF8.GetBytes(builder.Configuration["Jwt:Key"]!))
        };
    });


builder.Services.AddAuthorization();
builder.Services.AddControllers();

builder.Services.AddHostedService<PlanExpirationService>();

builder.Services.AddSingleton<IConnectionMultiplexer>(sp =>
    ConnectionMultiplexer.Connect(builder.Configuration.GetConnectionString("Redis")!));

builder.Services.AddSingleton<CacheService>();

builder.Services.AddHttpClient("PythonService", client =>
{
    client.BaseAddress = new Uri("http://127.0.0.1:8000");
    client.Timeout = TimeSpan.FromSeconds(30);
});

builder.Services.AddScoped<PythonServiceClient>();

// Rate limiting: una sola instancia del store compartida por el middleware y el cleanup
builder.Services.AddSingleton<RateLimitingCounterStore>();
builder.Services.AddSingleton<IRateLimitingCounterStore>(sp => sp.GetRequiredService<RateLimitingCounterStore>());
builder.Services.AddSingleton<EndpointRateLimitResolver>();
builder.Services.AddHostedService<RateLimitCleanUpService>();

// ! *******************************************
// ! ********** APP ******************
// ! *******************************************
var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

app.UseAuthentication();

app.UseMiddleware<RateLimitingMiddleware>();

app.UseAuthorization();

app.MapControllers();

app.Run();
