const express = require("express");
const purchasingController = require("../controllers/purchasingController");
const authController = require("../controllers/authController");
const deletePurchase = require("./../public/js/deletePurchase")

const router = express.Router();

router.use(authController.protect);

router
  .route("/checkout-session/:productId")
  .post(purchasingController.getCheckoutSession);

// router
//   .route("/user/:userId")
//   .post(purchasingController.getUser);

// router.use(authController.restrictTo('admin','seller'));

// router
//   .route('/')
//   .get(purchasingController.getAllPurchases)
//   .post(purchasingController.createPurchase);

// router.route('/createPurchase').post(purchasingController.createPurchase);
// router.route('/updatePurchase').post(purchasingController.updateMe);
// router.route('/deletePurchase').post(purchasingController.deletePurchase);

// router
//   .route('/:id')
//   .get(purchasingController.getPurchase)
//   .patch(purchasingController.updatePurchase)
//   .delete(deletePurchase);

module.exports = router;
