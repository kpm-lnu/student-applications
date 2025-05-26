function goToHomePage() {
    window.location.href = "../index.html";
}

var reviews = document.querySelectorAll('.review');
  var showMoreButton = document.querySelector('.showMoreButton');
  var hideButton = document.querySelector('.hideButton');

  function toggleReviews() {
    for (var i = 1; i < reviews.length; i++) {
      reviews[i].classList.toggle('review-hidden');
    }

    showMoreButton.style.display = 'none';
    hideButton.style.display = 'block';
  }

  function hideReviews() {
    for (var i = 1; i < reviews.length; i++) {
      reviews[i].classList.add('review-hidden');
    }

    hideButton.style.display = 'none';
    showMoreButton.style.display = 'block';
  }




  

  