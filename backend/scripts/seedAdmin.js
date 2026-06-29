/**
 * seedAdmin.js
 * 
 * Run once to create the default admin account:
 *   node backend/scripts/seedAdmin.js
 * 
 * Default credentials:
 *   Email:    admin@eyezora.com
 *   Password: Admin@123
 */

const path = require("path");
require("dotenv").config({ path: path.join(__dirname, "../.env") });

const dns = require("dns");
dns.setServers(["8.8.8.8", "8.8.4.4"]);

const mongoose = require("mongoose");
const bcrypt = require("bcryptjs");
const Admin = require("../models/Admin");

async function seed() {
  try {
    await mongoose.connect(process.env.MONGO_URI);
    console.log("✅ MongoDB connected");

    const existing = await Admin.findOne({ email: "admin@eyezora.com" });
    if (existing) {
      console.log("⚠️  Admin already exists. Skipping seed.");
      process.exit(0);
    }

    const passwordHash = await bcrypt.hash("Admin@123", 12);

    await Admin.create({
      name: "EyeZora Admin",
      email: "admin@eyezora.com",
      passwordHash,
    });

    console.log("✅ Admin created successfully!");
    console.log("   Email:    admin@eyezora.com");
    console.log("   Password: Admin@123");
    console.log("   ⚠️  Change this password immediately in production!");
    process.exit(0);
  } catch (err) {
    console.error("❌ Seed failed:", err.message);
    process.exit(1);
  }
}

seed();
