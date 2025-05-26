//scroll
document.addEventListener("DOMContentLoaded", function () {
  var scrollingTitle = document.querySelector(".scrolling-title");

  window.addEventListener("scroll", function () {
    if (window.scrollY > 0) {
      scrollingTitle.classList.add("fixed");
    } else {
      scrollingTitle.classList.remove("fixed");
    }
  });
});

var reviewsSection = document.getElementById('reviews');

function checkScrollPosition() {
  var scrollPosition = window.innerHeight + window.pageYOffset;
  var documentHeight = document.documentElement.scrollHeight;

  if (scrollPosition >= documentHeight) {
    reviewsSection.classList.add('show');
    window.removeEventListener('scroll', checkScrollPosition);
  }
}

function goToHomePage() {
  window.scrollTo(0, 0);
}

function goToCatalogPage() {
  window.location.href = "./catalog/catalog.html";
}

function goToContactsPage() {
  window.location.href = "./contacts/contacts.html";
}

window.addEventListener('scroll', checkScrollPosition);


var footerSection = document.querySelector(".footer-section");
var windowHeight = window.innerHeight;
var scrollCount = 0;

function checkFooterVisibility() {
  var footerTop = footerSection.getBoundingClientRect().top;

  if (footerTop <= windowHeight) {
    scrollCount++;
    if (scrollCount === 2) {
      window.removeEventListener("scroll", checkFooterVisibility);
    }
  }
}

window.addEventListener("scroll", checkFooterVisibility);


window.addEventListener('scroll', function () {
  var footer = document.getElementById('footer');
  var scrollPosition = window.innerHeight + window.pageYOffset;
  var documentHeight = document.documentElement.scrollHeight;

  if (scrollPosition >= documentHeight) {
    footer.style.display = 'block';
  } else {
    footer.style.display = 'none';
  }
});




$(document).ready(function () {
  $('.featured-products').slick({
    slidesToShow: 3,
    slidesToScroll: 1,
    autoplay: true,
    autoplaySpeed: 2000,
    infinite: true,
    prevArrow: '<button type="button" class="slick-prev">Previous</button>',
    nextArrow: '<button type="button" class="slick-next">Next</button>',
    responsive: [
      {
        breakpoint: 768,
        settings: {
          slidesToShow: 2
        }
      },
      {
        breakpoint: 480,
        settings: {
          slidesToShow: 1
        }
      }
    ]
  });
});





