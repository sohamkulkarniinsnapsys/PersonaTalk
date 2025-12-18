from django.test import TestCase, Client
from django.urls import reverse
from django.contrib.auth.models import User

class AuthTests(TestCase):
    def setUp(self):
        self.client = Client()
        self.user = User.objects.create_user(username='testuser', password='password123')

    def test_login_page_loads(self):
        """Root URL should load login page"""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'registration/login.html')

    def test_register_page_loads(self):
        """Register URL should load register page"""
        response = self.client.get('/accounts/register/')
        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, 'registration/register.html')

    def test_login_logic(self):
        """Valid login should redirect"""
        response = self.client.post('/', {'username': 'testuser', 'password': 'password123'})
        self.assertRedirects(response, '/api/', fetch_redirect_response=False) # Matches settings.LOGIN_REDIRECT_URL

    def test_register_logic(self):
        """Valid registration should create user and redirect to login"""
        response = self.client.post('/accounts/register/', {
            'username': 'newuser',
            'password1': 'StrongPassword123!',
            'password2': 'StrongPassword123!'
        })
        self.assertRedirects(response, reverse('login'), fetch_redirect_response=False)
        self.assertTrue(User.objects.filter(username='newuser').exists())

    def test_secure_redirect(self):
        """Login with next param should validate host"""
        # Internal redirect - Safe
        response = self.client.post('/?next=/api/safe', {
            'username': 'testuser', 
            'password': 'password123'
        })
        self.assertRedirects(response, '/api/safe', fetch_redirect_response=False)

        # External redirect - Unsafe (Should ignore and go to default)
        response = self.client.post('/?next=http://evil.com', {
            'username': 'testuser', 
            'password': 'password123'
        })
        self.assertRedirects(response, '/api/', fetch_redirect_response=False)
