document.addEventListener('DOMContentLoaded', function() {
    // Handle search form
    const searchForm = document.getElementById('searchForm');
    if (searchForm) {
        searchForm.addEventListener('submit', function(e) {
            e.preventDefault();
            const searchInput = document.getElementById('searchInput');
            window.location.href = `/search?q=${encodeURIComponent(searchInput.value)}`;
        });
    }

    // Handle loading states
    const showLoading = () => {
        const loadingEl = document.createElement('div');
        loadingEl.className = 'loading-spinner mx-auto';
        document.querySelector('.content-area').prepend(loadingEl);
    };

    const hideLoading = () => {
        const spinner = document.querySelector('.loading-spinner');
        if (spinner) spinner.remove();
    };

    // Handle movie card clicks
    const movieCards = document.querySelectorAll('.movie-card');
    movieCards.forEach(card => {
        card.addEventListener('click', function(e) {
            // Don't navigate if clicking the watch button
            if (e.target.classList.contains('watch-now-btn')) {
                e.preventDefault();
                return;
            }
            const movieId = this.dataset.movieId;
            window.location.href = `/movie/${movieId}`;
        });
    });
    // Handle watch now buttons
    const watchButtons = document.querySelectorAll('.watch-now-btn');
    watchButtons.forEach(button => {
        button.addEventListener('click', function(e) {
            e.preventDefault();
            const movieTitle = this.dataset.movieTitle;
            const searchQuery = `${movieTitle} movie watch online`;
            window.open(`https://www.google.com/search?q=${encodeURIComponent(searchQuery)}`, '_blank');
        });
    });

    // Handle filter changes
    const filterSelect = document.getElementById('genreFilter');
    if (filterSelect) {
        filterSelect.addEventListener('change', function() {
            showLoading();
            window.location.href = `/search?genre=${this.value}`;
        });
    }

    // Handle errors
    const showError = (message) => {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error-message';
        errorDiv.textContent = message;
        document.querySelector('.content-area').prepend(errorDiv);
    };
});
