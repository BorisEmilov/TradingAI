using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using ApiBackend.Api.Models;
using Microsoft.EntityFrameworkCore;


namespace ApiBackend.Api.Data
{
    public class ApplicationDBContext : DbContext
    {
        public ApplicationDBContext(DbContextOptions dbContextOptions) : base(dbContextOptions)
        {
            
        }

        public DbSet<User> User { get; set; }
        public DbSet<RefreshDoc> RefreshDoc { get; set; }
        public DbSet<Plan> Plan { get; set; }
        public DbSet<MtAccount> MtAccount {get; set;}

        protected override void OnModelCreating(ModelBuilder modelBuilder)
        {
            base.OnModelCreating(modelBuilder);

            // User tiene DOS relaciones con Plan (actual y pendiente): EF no puede adivinar
            // a cuál pertenece Plan.Users -> se declara explícitamente.
            modelBuilder.Entity<User>()
                .HasOne(u => u.Plan)
                .WithMany(p => p.Users)
                .HasForeignKey(u => u.PlanId);

            modelBuilder.Entity<User>()
                .HasOne(u => u.PendingPlan)
                .WithMany()
                .HasForeignKey(u => u.PendingPlanId);
        }
    }
}
