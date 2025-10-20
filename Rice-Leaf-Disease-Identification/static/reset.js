document.addEventListener('DOMContentLoaded', (event) => {
    const resetButton = document.getElementById('resetButton');
    if (resetButton) {
        resetButton.addEventListener('click', () => {
            window.location.href = '/';
        });
    }
});