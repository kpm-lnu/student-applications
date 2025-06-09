const mongoose = require('mongoose');

const purchasingSchema = new mongoose.Schema({
    product:{
        type: mongoose.Schema.ObjectId,
        ref: 'Product',
        required: [true, 'Purchase must belong to some product']
    },
    user:{
        type: mongoose.Schema.ObjectId,
        ref: 'User',
        required: [true, 'User must belong to some product']
    },
    price:{
        type: Number,
        required: [true, 'Purchase must have a price']
    },
    createdAt:{
        type: Date,
        default: Date.now()   
    },
    paid: {
        type: Boolean,
        default: true
    }
});

purchasingSchema.pre(/^find/,function(next) {
    this.populate('user').populate({
        path:'product',
        select: 'name'
    });
    next();
})

const Purchasing = mongoose.model('Purchasing', purchasingSchema);

module.exports = Purchasing;