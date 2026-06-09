function checkAuth() {
    const token = localStorage.getItem('leukonet_token');
    if (!token && window.location.pathname !== '/login') {
        window.location.href = '/login';
    }
}

function logout() {
    localStorage.removeItem('leukonet_token');
    window.location.href = '/login';
}
