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
        public DbSet<MtAccount> MtAccount { get; set; }
        public DbSet<Plan> Plan { get; set; }
    }
}
