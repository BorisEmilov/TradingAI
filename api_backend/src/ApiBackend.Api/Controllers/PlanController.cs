using System;
using ApiBackend.Api.Dtos.Plan;
using ApiBackend.Api.Mappers;
using ApiBackend.Api.Models;
using ApiBackend.Api.Interfaces;
using Microsoft.AspNetCore.Authorization;
using Microsoft.AspNetCore.Http.HttpResults;
using Microsoft.AspNetCore.Mvc;



namespace ApiBackend.Api.Controllers
{
    [ApiController]
    [Route("api/")]
    public class PlanController : ControllerBase
    {
        private readonly IPlanRepository _planRepo;
        public PlanController(IPlanRepository planRepo)
        {
            _planRepo = planRepo;
        }

        [Authorize(Roles = nameof(Roles.ADMIN))]
        [HttpPost]
        [Route("new-plan")]
        public async Task<IActionResult> CreatePlan([FromBody] CreatePlanDto dto)
        {
            var response = await _planRepo.CreatePlan(dto.FromCreatePlanDtoToPlan());
            if(response == null)
            {
                return BadRequest();
            }
            return Ok(response.FromPlanToGetPlanDto());
        }

        [Authorize(Roles = nameof(Roles.ADMIN))]
        [HttpGet]
        [Route("plans/{planId}")]
        public async Task<IActionResult> GetPlanById([FromRoute] Guid planId)
        {
            var response = await _planRepo.GetPlanById(planId);
            if(response == null)
            {
                return NotFound("Plan not found");
            }
            return Ok(response.FromPlanToGetPlanDto());
        }

        [Authorize(Roles = nameof(Roles.ADMIN))]
        [HttpGet]
        [Route("plans/users/{planId}")]
        public async Task<IActionResult> GetUsersPlan(Guid planId)
        {
            var response = await _planRepo.GetUsersPlan(planId);
            if(response == null)
            {
                return NotFound("No plan or users found");
            }
            return Ok(new
            {
                plan = response.Value.Item1.FromPlanToGetPlanDto(),
                usersCount = response.Value.Item2.Count(),
                users = response.Value.Item2.Select(u => u.FromUserToGetUserDto()),
            });
        }

        [Authorize(Roles = nameof(Roles.ADMIN))]
        [HttpPut]
        [Route("plans/update/{planId}")]
        public async Task<IActionResult> UpdatePlan(
            [FromRoute] Guid planId,
            [FromBody] UpdatePlanDto dto
        )
        {
            var response = await _planRepo.UpdatePlan(planId, dto.FromUpdatePlanDtoToPlan());
            if(response == null)
            {
                return BadRequest("Error updating plan");
            }
            return Ok(response.FromPlanToGetPlanDto());
        }


        [Authorize(Roles = nameof(Roles.ADMIN))]
        [HttpDelete]
        [Route("plans/delete/{planId}")]
        public async Task<IActionResult> DeletePlan([FromRoute] Guid planId)
        {
            var response = await _planRepo.DeletePlan(planId);
            if(response == false)
            {
                return BadRequest("Plan not found or still assigned to users");
            }
            return Ok("Successfuly deleted");
        }
    }
}
