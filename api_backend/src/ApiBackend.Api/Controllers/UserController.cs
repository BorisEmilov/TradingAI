using System.Security.Claims;
using ApiBackend.Api.Dtos.User;
using ApiBackend.Api.Mappers;
using ApiBackend.Api.RateLimiting;
using ApiBackend.Api.Repositories;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Mvc;

namespace ApiBackend.Api.Controllers;

[ApiController]
[Route("api/")]
public class UserController : ControllerBase
{
    private readonly UserRepository _userRepo;
    public UserController(UserRepository userRepo)
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
    public async Task<IActionResult> Login([FromBody] string email, [FromBody] string password)
    {
        if(email == null)
        {
            return BadRequest("Enter valid email");
        }
        if(password == null)
        {
            return BadRequest("Enter valid password");
        }

        var result =  await _userRepo.Login(email, password);
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
            Expires = DateTime.UtcNow.AddDays(7)
        });

        return Ok(new {User = user.FromUserToGetUserDto(), accessToken = AccessToken});
    }

    [HttpPost]
    [Authorize]
    [RateLimit(limitRequests: 30, windowSeconds: 60)]
    [Route("refresh")]
    public async Task<IActionResult> Refresh()
    {
        var refreshToken = Request.Cookies["refreshToken"];
        if (string.IsNullOrEmpty(refreshToken))
        {
            return Unauthorized();
        }

        var response = _userRepo.RefreshAsync(refreshToken);
        if(response == null)
        {
            return Unauthorized();
        }

        return Ok(new { AccessToken = response });
    }

    
    [HttpGet]
    [Authorize]
    [RateLimit(limitRequests: 1, windowSeconds: 60)]
    [Route("/send-verification-link")]
    public async Task<IActionResult> SendVerificationLink()
    {
        var response = await _userRepo.SendVerificationLink(Guid.Parse(ClaimTypes.NameIdentifier));
        if(response == null)
        {
            return Unauthorized();
        }
        return Ok(new { msg = response });
    }


    [HttpPatch]
    [Route("/verify/{token}")]
    public async Task<IActionResult> VerifyEmail(
        [FromRoute] string token
    )
    {
        var response = await _userRepo.VerifyEmail(Guid.Parse(ClaimTypes.NameIdentifier), token);
        if(response == null)
        {
            return BadRequest();
        }

        return Ok(new { msg = response });
    }

    [HttpGet]
    [Authorize]
    [Route("my-profile")]
    public async Task<IActionResult> getMyProfile()
    {
        var user = await _userRepo.GetUserById(Guid.Parse(ClaimTypes.NameIdentifier));
        if(user == null)
        {
            return NotFound();
        }
        return Ok(user.FromUserToGetUserDto());
    }

    [HttpPatch]
    [Authorize]
    [RateLimit(limitRequests: 1, windowSeconds: 60)]
    [Route("password-update/send-link")]
    public async Task<IActionResult> SendUpdatePasswordLink()
    {
        var response = await _userRepo.SendChangePasswordLink(Guid.Parse(ClaimTypes.NameIdentifier));
        if(response == null)
        {
            return BadRequest();
        }
        return Ok(new { msg = response });
    }

    [HttpPatch]
    [Authorize]
    [Route("change-password/{token}")]
    public async Task<IActionResult> UpdatePassword(
        [FromRoute] string token,
        [FromBody] UpdatePasswordDto dto
    )
    {
        if(dto.Password != dto.RepeatPassword)
        {
            return BadRequest("Password do not match");
        }

        string hashedPassword = BCrypt.Net.BCrypt.HashPassword(dto.Password);

        string response = await _userRepo.ChangePassword(Guid.Parse(ClaimTypes.NameIdentifier), hashedPassword, token);

        return Ok(new { msg = response });
    }

    [HttpPatch]
    [Authorize]
    [Route("my-profile/delete")]
    public async Task<IActionResult> DeleteProfile()
    {
        var response = await _userRepo.DeleteUser(Guid.Parse(ClaimTypes.NameIdentifier));
        if(response == false)
        {
            return BadRequest();
        }
        return Ok("Successfuly deleted account");
    }
}
