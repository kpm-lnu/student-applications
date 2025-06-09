const Product = require("./../models/productModel");
const Purchases = require("./../models/purchasingModel");
const User = require("./../models/userModel");
const Reviews = require('./../models/reviewModel');
const catchAsync = require("./../utils/catchAsync");
const AppError = require("./../utils/appError");
const purchasingController = require('./../controllers/purchasingController');
const { CountryCodes } = require("validator/lib/isISO31661Alpha2");

exports.getOverview = catchAsync(async (req, res) => {
  // get product data from collection
  const products = await Product.find();

  // build template
  // render with data
  res.status(200).render("overview", {
    title: "All products",
    products,
  });
});

exports.getProduct = catchAsync(async (req, res, next) => {
  //get the data from the requested tour
  const product = await Product.findOne({ slug: req.params.slug }).populate({
    path: "reviews",
    fields: "review rating user",
  });

  if (!product) {
    return next(new AppError("There is no product with this name", 404));
  }

  //build template
  //render using data
  res.status(200).render("product", {
    title: product.name,
    product,
  });
});

exports.getLoginForm = catchAsync(async (req, res) => {
  res.status(200).render("login", {
    title: "Log into your account",
  });
});

exports.getSignForm = catchAsync(async (req, res) => {
  res.status(200).render("signup", {
    title: "Sign up your account",
  });
});

exports.getAccount = async(req, res) => {
  const timestamp = Date.now();
  const dateObj = new Date(timestamp);
  let month   = dateObj.getUTCMonth() + 1; // months from 1-12
  let monthE = '';
  let day     = dateObj.getUTCDate();
  let dayE    = '';
  const year    = dateObj.getUTCFullYear();

  // Using padded values, so that 2023/1/7 becomes 2023/01/07
  // const pMonth        = Number(month.toString().padStart(2, "0"));
  // const pMonthE        = Number(monthE.toString().padStart(2, "0"));
  // const pDay          = Number(day.toString().padStart(2, "0"));
  // const pDayE         = Number(dayE.toString().padStart(2, "0"));
  // const newPaddedDate = `${year}/${pMonth}/${pDay}`;
  
  const purchasesYear = await Purchases.aggregate([
    {
        $match: {
          createdAt: {
                $gte: new Date(`${year}-01-01T00:00:00.000Z`), // Start of the year
                $lt: new Date(`${year+1}-01-01T00:00:00.000Z`) // Start of the next year
            }
        }
    }
  ]);
  // let purchasesMonth = '';
  if(month < 10){
    month += 1;
    monthE = '0' + month;
    month -= 1;
    month = '0' + month;  
  }else{
    monthE = month+1; 
  } 

  const purchasesMonth = await Purchases.aggregate([
      {
        $match: {
          createdAt: {
                $gte: new Date(`${year}-${month}-01T00:00:00.000Z`), // Start of the year
                $lt: new Date(`${year}-${monthE}-01T00:00:00.000Z`) // Start of the next year
            }
        }
    }
    ]);

  // let purchasesDay = '';
  if(day < 10){
    day += 1;
    dayE = '0' + day;
    day -= 1;
    day = '0' + day; 
  }else{
    dayE = day+1;
  }

  const purchasesDay = await Purchases.aggregate([
      {
        $match: {
          createdAt: {
                $gte: new Date(`${year}-${month}-${day}T00:00:00.000Z`), // Start of the year
                $lt: new Date(`${year}-${monthE}-${dayE}T00:00:00.000Z`) // Start of the next year
            }
        }
      }
    ]);

  //year
  const pY = purchasesYear.map(el => el.price);
  const pYP = pY.reduce((accumulator, currentValue) => accumulator + currentValue, 0);
  const countY = pY.length;
  // //month
  const pM = purchasesMonth.map(el => el.price);
  const pMP = pM.reduce((accumulator, currentValue) => accumulator + currentValue,0);
  const countM = pM.length;
  //day
  const pD = purchasesDay.map(el => el.price);
  const pDP = pD.reduce((accumulator, currentValue) => accumulator + currentValue,0);
  const countD = pD.length;

  res.status(200).render("account", {
    title: "Your account",
      pYP,
      countY,
      pMP,
      countM,
      pDP,
      countD
  });

  // res.status(200).json({
  //   status: "success",
  //   data:{
  //     pYP,
  //     countY,
  //     pMP,
  //     countM,
  //     pDP,
  //     countD
  //   }
  //   // Include other data (pMP, countM, pDP, countD) if available
  // });

};

exports.getYearlyPlan = catchAsync(async (req,res,next) => {
  const year = req.params.year * 1;
  const month = req.params.month * 1;
  const monthE = month

  const plan = await Purchases.aggregate([
    {
      $match:{
        createdAt: {
              $gte: new Date(`${year}-01T00:00:00.000Z`), // Start of the year
              $lt: new Date(`${year+1}-01-01T00:00:00.000Z`) // Start of the next year
          }
      }
    }
  ]);

  res.status(200).json({
    status: 'success',
    data: {
      plan
    }
  });

});

exports.getMonthlyPlan = catchAsync(async (req,res,next) => {
  const year = req.params.year * 1;
  const month = req.params.month * 1;
  const day = req.params.day * 1;

  const plan = await Purchases.aggregate([
    {
      $match:{
        createdAt: {
              $gte: new Date(`${year}-${month}-01T00:00:00.000Z`), // Start of the year
              $lt: new Date(`${year}-${month}-01T00:00:00.000Z`) // Start of the next year
          }
      }
    }
  ]);

  res.status(200).json({
    status: 'success',
    data: {
      plan
    }
  });

});

exports.getDailyPlan = catchAsync(async (req,res,next) => {
  const year = req.params.year * 1;
  const month = req.params.month * 1;
  const day = req.params.day * 1;

  const plan = await Purchases.aggregate([
    {
      $match:{
        createdAt: {
              $gte: new Date(`${year}-${month}-${day}T00:00:00.000Z`), // Start of the year
              $lt: new Date(`${year}-${month}-${day}T00:00:00.000Z`) // Start of the next year
          }
      }
    }
  ]);

  res.status(200).json({
    status: 'success',
    data: {
      plan
    }
  });

});

exports.getMyProducts = catchAsync(async (req,res) => {
  const purchases = await Purchases.find({ user:req.user.id});
  //find products with ids from purchases
  const productIDs = purchases.map(el => el.product);
  // select all the products that are in the productsIDs
  const products = await Product.find({ _id: {$in: productIDs}});
  const prices = products.map(el => el.price);
  const totalPrice = prices.reduce((acc, curr) => acc + curr, 0);

  res.status(200).render('overview', {
    title: 'My products',
    products,
    totalPrice
  })
});

exports.getMyReviews = catchAsync(async (req,res) => {
  const reviews = await Reviews.find({ user:req.user.id}).populate({
    path: "product",
    fields: "photo name",
  });

  res.status(200).render('reviews', {
    title: 'My products',
    reviews
  })
  // res.status(200).json({
  //   status:'success',
  //   data: {
  //     reviews
  //   }
  // })
  
});

exports.updateUserData = catchAsync(async (req, res) => {
  const updatedUser = await User.findByIdAndUpdate(
    req.user.id,
    {
      name: req.body.name,
      email: req.body.email,
    },
    {
      new: true,
      runValidators: true,
    },
  );

  res.status(200).render("account", {
    title: "Your account",
    user: updatedUser,
  });
});

exports.manageProducts = catchAsync(async (req, res) => {
  res.status(200).render('manageProducts', {
    title: 'Managing Products'
  })
});

exports.manageUsers = catchAsync(async (req, res) => {
  res.status(200).render('manageUsers', {
    title: 'Managing User'
  })
});

exports.managePurchases = catchAsync(async (req, res) => {
  res.status(200).render('managePurchases', {
    title: 'Managing Purchases'
  })
});