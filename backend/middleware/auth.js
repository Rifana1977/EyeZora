const jwt = require("jsonwebtoken");

const JWT_SECRET = process.env.JWT_SECRET || "eyezora_secret_change_in_prod";

/**
 * Verifies JWT from Authorization header.
 * Attaches decoded payload to req.user.
 */
const verifyToken = (req, res, next) => {
  const authHeader = req.headers["authorization"];
  if (!authHeader || !authHeader.startsWith("Bearer ")) {
    return res.status(401).json({ error: "No token provided" });
  }

  const token = authHeader.split(" ")[1];
  try {
    const decoded = jwt.verify(token, JWT_SECRET);
    req.user = decoded;
    next();
  } catch (err) {
    return res.status(401).json({ error: "Invalid or expired token" });
  }
};

/**
 * Ensures the authenticated user is an admin.
 * Must be used AFTER verifyToken.
 */
const requireAdmin = (req, res, next) => {
  if (!req.user || req.user.role !== "admin") {
    return res
      .status(403)
      .json({ error: "Access denied. Admin role required." });
  }
  next();
};

/**
 * Ensures the authenticated user is a student.
 * Must be used AFTER verifyToken.
 */
const requireStudent = (req, res, next) => {
  if (!req.user || req.user.role !== "student") {
    return res
      .status(403)
      .json({ error: "Access denied. Student role required." });
  }
  next();
};

module.exports = { verifyToken, requireAdmin, requireStudent, JWT_SECRET };
