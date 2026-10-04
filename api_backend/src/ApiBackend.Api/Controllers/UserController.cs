using System.Security.Claims;
using ApiBackend.Api.Dtos.User;
using ApiBackend.Api.Mappers;
using ApiBackend.Api.RateLimiting;
using ApiBackend.Api.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ApiBackend.Api.Controllers;

[ApiController]
[Route("api/")]
public class UserController : ControllerBase
{
    private readonly IUserRepository _userRepo;
    public UserController(IUserRepository userRepo)
    {
        _userRepo = userRepo;
    }


    [HttpPost]
    [RateLimit(limitRequests: 5, windowSeconds: 60)]
    [Route("register")]
    public async Task<IActionResult> CreateUser([FromBody] NewUserDto dto)
    {
        if (dto.Password != dto.RepeatPassword)
        {
            return BadRequest("Password do not match");    
        }
        string hashedPassword = BCrypt.Net.BCrypt.HashPassword(dto.Password);

        CreateUserDto userDto = new CreateUserDto
        {
            Email = dto.Email,
            FullName = dto.FullName,
            HashedPassword = hashedPassword,
        };

        var result = await _userRepo.CreateUser(userDto.FromCreateUserDtoToUser());
        if(result == null)
        {
            return BadRequest("Emal alredy exists");
        }

        return Ok(new
        {
            msg = result.Value.Item2,
            user = result.Value.Item1.FromUserToGetUserDto(),
        });
    }


    [HttpPost]
    [RateLimit(limitRequests: 10, windowSeconds: 60)]
    [Route("login")]
    public async Task<IActionResult> Login([FromBody] LoginDto dto)
    {
        if(string.IsNullOrWhiteSpace(dto.Email))
        {
            return BadRequest("Enter valid email");
        }
        if(string.IsNullOrWhiteSpace(dto.Password))
        {
            return BadRequest("Enter valid password");
        }

        var result =  await _userRepo.Login(dto.Email, dto.Password);
        if(result == null)
        {
            return BadRequest();
        }

        var (user, AccessToken, RefreshToken, ExpirityDate) = result.Value;

        Response.Cookies.Append("refreshToken", RefreshToken, new CookieOptions
        {
            HttpOnly = true,
            Secure = true,
            SameSite = SameSiteMode.Strict,
            Expires = ExpirityDate
        });

        return Ok(new {User = user.FromUserToGetUserDto(), accessToken = AccessToken});
    }

    [HttpPost]
    [RateLimit(limitRequests: 30, windowSeconds: 60)]
    [Route("refresh")]
    public async Task<IActionResult> Refresh()
    {
        var refreshToken = Request.Cookies["refreshToken"];
        if (string.IsNullOrEmpty(refreshToken))
        {
            return Unauthorized();
        }

        var response = await _userRepo.RefreshAsync(refreshToken);
        if(response == null)
        {
            return Unauthorized();
        }

        var (accessToken, newRefreshToken, expiresAt) = response.Value;
        Response.Cookies.Append("refreshToken", newRefreshToken, new CookieOptions
        {
            HttpOnly = true,
            Secure = true,
            SameSite = SameSiteMode.Strict,
            Expires = expiresAt
        });

        return Ok(new { AccessToken = accessToken });
    }

    
    [HttpPost]
    [Authorize]
    [RateLimit(limitRequests: 1, windowSeconds: 60)]
    [Route("send-verification-code")]
    public async Task<IActionResult> SendVerificationCode()
    {
        var response = await _userRepo.SendVerificationCode(Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!));
        if(response == null)
        {
            return BadRequest("User not found or already verified");
        }
        return Ok(new { msg = response });
    }


    [HttpPost]
    [Authorize]
    [RateLimit(limitRequests: 5, windowSeconds: 60)]
    [Route("verify-email")]
    public async Task<IActionResult> VerifyEmail([FromBody] VerifyCodeDto dto)
    {
        var response = await _userRepo.VerifyEmail(Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!), dto.Code);
        if(response == null)
        {
            return BadRequest("Invalid or expired code");
        }

        return Ok(new { msg = response });
    }

    [HttpGet]
    [Authorize]
    [Route("my-profile")]
    public async Task<IActionResult> getMyProfile()
    {
        var user = await _userRepo.GetUserById(Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!));
        if(user == null)
        {
            return NotFound();
        }
        return Ok(user.FromUserToGetUserDto());
    }

    [HttpPost]
    [Authorize]
    [RateLimit(limitRequests: 1, windowSeconds: 60)]
    [Route("password-update/send-code")]
    public async Task<IActionResult> SendUpdatePasswordCode()
    {
        var response = await _userRepo.SendChangePasswordCode(Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!));
        if(response == null)
        {
            return BadRequest();
        }
        return Ok(new { msg = response });
    }

    [HttpPatch]
    [Authorize]
    [RateLimit(limitRequests: 5, windowSeconds: 60)]
    [Route("change-password")]
    public async Task<IActionResult> UpdatePassword([FromBody] UpdatePasswordDto dto)
    {
        if(dto.Password != dto.RepeatPassword)
        {
            return BadRequest("Password do not match");
        }

        string hashedPassword = BCrypt.Net.BCrypt.HashPassword(dto.Password);

        var response = await _userRepo.ChangePassword(Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!), hashedPassword, dto.Code);
        if(response == null)
        {
            return BadRequest("Invalid or expired code");
        }

        return Ok(new { msg = response });
    }

    [HttpDelete]
    [Authorize]
    [Route("my-profile/delete")]
    public async Task<IActionResult> DeleteProfile()
    {
        var response = await _userRepo.DeleteUser(Guid.Parse(User.FindFirstValue(ClaimTypes.NameIdentifier)!));
        if(response == false)
        {
            return BadRequest();
        }
        return Ok("Successfuly deleted account");
    }
}
