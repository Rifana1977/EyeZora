const express = require("express");
const router = express.Router();
const { verifyToken, requireAdmin } = require("../middleware/auth");
const { addQuestion } = require("../controllers/adminController");

// Admin-only route to add a question (kept for backward compatibility)
router.post("/", verifyToken, requireAdmin, addQuestion);

module.exports = router;