const catchAsync = require("../utils/catchAsync");
const AppError = require("../utils/appError");
const factory = require("./handlerFactory");
const Product = require("../models/productModel");
const Purchase = require("../models/purchasingModel");
const stripe = require("stripe")(process.env.STRIPE_SECRET_KEY);

exports.getCheckoutSession = catchAsync(async (req, res, next) => {
  try{  
      const product = await Product.findById(req.params.productId);

      const session = await stripe.checkout.sessions.create({
        payment_method_types: ["card"],
        success_url: `${req.protocol}://${req.get("host")}/?product=${req.params.productId}&user=${req.user.id}&price=${product.price}`,
        cancel_url: `${req.protocol}://${req.get("host")}/product/${product.slug}`,
        customer_email: req.user.email,
        client_reference_id: req.params.productId,
        mode: "payment",
        line_items: [
          {
            price_data: {
              currency: "usd",
              product_data: {
                name: product.name,
                description: product.summary,
              },
              unit_amount:Math.round(product.price * 100),
            },
            quantity: 1,
          },
        ]
    });
    
    res.status(200).json({
      status: "success",
      session
    });
  }catch(err){
    // console.log(err);
    showAlert('happens',500);
    return next(new AppError(err,err.statusCode));
  }
  
});

exports.createPurchaseCheckout = catchAsync(async (req,res,next) => {
  const { product, user, price } = req.query;
  // console.log(product,"|",user,"|",price);

  if (!product && !user && !price) return next();
  await Purchase.create({ product, user, price });
  // console.log(req.originalUrl.split('?')[0]);
  res.redirect(req.originalUrl.split('?')[0]);
});

exports.updateMe = catchAsync(async (req, res, next) => {
  //filtered unwanted fields for updating
  //update user document
  const updatedPurchase = await Purchase.findByIdAndUpdate(req.purchase.id, req.body, {
    new: true,
    runValidators: true,
  });

  res.status(200).json({
    status: "success",
    data: {
      purchase: updatedPurchase,
    },
  });
});

// exports.getUserStats = catchAsync(async (req,res,next) => {
//   const purchases = await Purchase.find({user: req.user.id});
//   const productsIDs = purchases.map(el => el.products);
//   const products = await Product.find({_id:{$in: productsIDs}});
//   const prices = products.map(el => el.price);
//   const totalPrice = prices.reduce((acc, curr) => acc + curr, 0);
//   res.status(200).render('overview', {
//     title: 'User products',
//     products,
//     totalPrice
//   })

// });

exports.createPurchase = factory.createOne(Purchase);
exports.getPurchase = factory.getOne(Purchase);
exports.getAllPurchases = factory.getAll(Purchase);
exports.updatePurchase = factory.updateOne(Purchase);
exports.deletePurchase = factory.deleteOne(Purchase);