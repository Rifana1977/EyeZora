const express = require("express");
const mongoose = require("mongoose");
const cors = require("cors");
require("dotenv").config();

// 👇 import routes
const questionRoutes = require("./routes/questionRoutes");

// 👇 create app
const app = express();

// 👇 middlewares
app.use(cors());
app.use(express.json());

// 👇 routes
app.use("/api/questions", require("./routes/questionRoutes"));

// 👇 test route
app.get("/", (req, res) => {
  res.send("Backend is running");
});

// 👇 MongoDB connection (ONLY ONCE)
mongoose
  .connect(process.env.MONGO_URI)
  .then(() => console.log("database connected successfully"))
  .catch(err => console.log("database connection failed", err));

// 👇 start server
const PORT = 5000;
app.listen(PORT, () => {
  console.log("server listening on port", PORT);
});
