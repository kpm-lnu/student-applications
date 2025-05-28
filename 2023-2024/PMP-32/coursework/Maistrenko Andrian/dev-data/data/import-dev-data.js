const fs = require("fs");
const mongoose = require("mongoose");
const dotenv = require("dotenv");
const Product = require("./../../models/productModel");
const User = require("./../../models/userModel");
const Review = require("./../../models/reviewModel");

dotenv.config({ path: "./config.env" });

const DB = process.env.DATABASE.replace(
  "<PASSWORD>",
  process.env.DATABASE_PASSWORD,
);

mongoose
  .connect(DB, {
    useNewUrlParser: true,
    useCreateIndex: true,
    useFindAnModify: false,
    useUnifiedTopology: true,
  })
  .then(() => console.log("DB connection successfully"));

// read json file

//const products = JSON.parse(fs.readFileSync(`${__dirname}/products.json`, 'utf-8'));
//const users = JSON.parse(fs.readFileSync(`${__dirname}/users.json`, 'utf-8'));
const reviews = JSON.parse(
  fs.readFileSync(`${__dirname}/reviews.json`, "utf-8"),
);

// import data into db

const importData = async () => {
  try {
    //await Product.create(products);
    //await User.create(users);
    await Review.create(reviews);

    console.log("Data imported successfully");
    process.exit();
  } catch (err) {
    console.log(err);
  }
};

// delete data from db

const deleteData = async () => {
  try {
    await Product.deleteMany();
    await User.deleteMany();
    await Review.deleteMany();

    console.log("Data deleted successfully");
    process.exit();
  } catch (err) {
    console.log(err);
  }
};

if (process.argv[2] === "--import") {
  importData();
} else if (process.argv[2] === "--delete") {
  deleteData();
}
