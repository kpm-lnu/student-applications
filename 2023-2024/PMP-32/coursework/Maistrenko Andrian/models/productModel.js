const { decodeBase64 } = require("bcryptjs");
const slugify = require("slugify");
const mongoose = require("mongoose");
const validator = require("validator");

const productSchema = new mongoose.Schema(
  {
    ratingsAverage: {
      type: Number,
      default: 4.5,
      min: [1, "Ratings must be above 1.0"],
      max: [5, "Ratings must be below 5.0"],
      set: (val) => Math.round(val * 10) / 10,
    },
    ratingsQuantity: {
      type: Number,
      default: 0,
      //validate: validator.isInt
    },
    imageCover: {
      type: String,
      required: [true, "A product must have a cover image"],
    },
    images: [String],
    createdAt: {
      type: Date,
      default: Date.now(),
      select: false,
    },
    name: {
      type: String,
      required: [true, "A product must have a name"],
      unique: true,
      trim: true,
      maxlength: [40, "A product must have less or equal than 40 characters"],
      minlength: [10, "A product must have more or equal than 10 characters"],
    },
    category: {
      type: String,
      required: [true, "A product must have a category"],
      enum: {
        values: ["Laptop", "Desktop", "Tablet", "Audio"],
        message: "Category is either Laptop, Desktop or Tablet",
      },
    },
    price: {
      type: Number,
      required: [true, "A product must have a price"],
    },
    summary: {
      type: String,
      trim: true,
    },
    description: {
      type: String,
      trim: true,
      required: [true, "A product must have a description"],
    },
    slug: {
      type: String,
    },
    processor: {
      type: String,
      //required: [true,'A product must have a processor']
    },
    graphicsCard: {
      type: String,
      //required: [true,'A product must have a graphics card']
    },
    ram: {
      type: String,
      //required: [true,'A product must have a ram']
    },
    drive: {
      type: String,
      //required: [true,'A product must have a drive']
    },
    warranty: {
      type: String,
      required: [true, "A product must have a warranty"],
    },
    priceDiscount: {
      type: Number,
      default: 0,
      validate: {
        validator: function (val) {
          // this only points to current doc on NEW document creation
          return val < this.price;
        },
        message: "Discount price ({VALUE}) should be below regular price",
      },
    },
    // sold:{
    //     type: Number,
    //     default: 0
    // },
    secretProduct: {
      type: Boolean,
      default: false,
    },
  },
  {
    toJSON: { virtuals: true },
    toObject: { virtuals: true },
  },
);

productSchema.index({ price: 1, ratingsAverage: -1 });
productSchema.index({ slug: 1 });

// virtual populate
productSchema.virtual("reviews", {
  ref: "Review",
  foreignField: "product",
  localField: "_id",
});

// document middleware that runs before .save() and .create() ONLY!
productSchema.pre("save", function (next) {
  this.slug = slugify(this.name, { lower: true });
  next();
});

// productSchema.post('save', function(doc,next) {
//     console.log(doc);
//     next();
// });

// query middleware

//productSchema.pre( 'find', function(next){
productSchema.pre(/^find/, function (next) {
  this.find({ secretProduct: { $ne: true } });

  this.start = Date.now();
  next();
});

// productSchema.post(/^find/, function (docs, next) {
//   console.log(`Query took ${Date.now() - this.start} milliseconds`);
//   next();
// });

// aggregation middleware

productSchema.pre(/^aggregate/, function (next) {
  this.pipeline().unshift({ $match: { secretProduct: { $ne: true } } });
  next();
});

const Product = mongoose.model("Product", productSchema);

module.exports = Product;
