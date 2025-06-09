const express = require("express");
const productController = require("./../controllers/productController");
const authController = require("./../controllers/authController");
const reviewRouter = require("./../routes/reviewRoutes");

const router = express.Router();

// router.param('id', productController.checkId)

router.use("/:productId/reviews", reviewRouter);

router
  .route("/top-5-products")
  .get(productController.aliasTopProducts, productController.getAllProducts);

router.route("/product-stats").get(productController.getProductStats);

router
  .route("/monthly-plan/:year")
  .get(
    authController.protect,
    authController.restrictTo("admin,seller"),
    productController.getMonthlyPlan,
  );

router
  .route("/")
  .get(productController.getAllProducts)
  .post(
    authController.protect,
    authController.restrictTo("admin"),
    productController.createProduct,
  );

router
  .route("/:id")
  .get(productController.getProduct)
  .patch(
    authController.protect,
    authController.restrictTo("admin", "seller"),
    productController.uploadProductImages,
    productController.resizeProductImages,
    productController.updateProduct,
  )
  .delete(
    authController.protect,
    authController.restrictTo("admin", "seller"),
    productController.deleteProduct,
  );

module.exports = router;
