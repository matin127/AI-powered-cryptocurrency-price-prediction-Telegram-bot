CryptoPredictor یک ربات تلگرامی پیشرفته است که از مدل‌های هوش مصنوعی (LSTM) برای پیش‌بینی قیمت ارزهای دیجیتال استفاده می‌کند. این ربات با استفاده از داده‌های واقعی صرافی KuCoin و الگوریتم‌های پیشرفته یادگیری ماشین، به شما کمک می‌کند تا قیمت‌های آینده ارزهای دیجیتال را پیش‌بینی کنید.

ویژگی‌ها
📈 پیش‌بینی قیمت ارز دیجیتال: پیش‌بینی دقیق قیمت‌های آینده برای ارزهای دیجیتال بر اساس داده‌های تاریخی.
⏱️ پشتیبانی از بازه‌های زمانی مختلف: قابلیت پیش‌بینی در بازه‌های زمانی مانند 1 ساعت، 1 روز، 1 هفته و غیره.
📊 داده‌های واقعی و به‌روز: استفاده از داده‌های OHLCV از صرافی KuCoin برای بهبود دقت پیش‌بینی.
🤖 یکپارچگی با تلگرام: تعامل مستقیم با ربات تلگرامی برای دریافت پیش‌بینی‌ها.
نحوه استفاده
شروع ربات: برای شروع، تنها کافیست دستور /start را ارسال کنید.

درخواست پیش‌بینی قیمت: دستور /predict SYMBOL TIMEFRAME را وارد کنید. به‌عنوان مثال:

plaintext
Copy code
/predict BTC/USDT 1h
ربات، پیش‌بینی قیمت برای جفت ارز BTC/USDT را در بازه زمانی 1 ساعت برای شما ارسال خواهد کرد.

نحوه نصب و راه‌اندازی
1. کلون کردن پروژه
ابتدا پروژه را از گیت‌هاب کلون کنید:

bash
Copy code
git clone https://github.com/yourusername/CryptoPredictor.git
cd CryptoPredictor
2. نصب وابستگی‌ها
تمامی وابستگی‌ها را با دستور زیر نصب کنید:

bash
Copy code
pip install ccxt python-telegram-bot pandas numpy matplotlib tensorflow
3. تنظیم توکن تلگرام
در فایل bot.py، توکن ربات تلگرام خود را وارد کنید:

python
Copy code
TOKEN = "YOUR_TELEGRAM_BOT_TOKEN"
4. اجرای ربات
پس از تنظیم توکن، ربات را با دستور زیر اجرا کنید:

bash
Copy code
python bot.py
حالا ربات آماده است تا شما را در پیش‌بینی قیمت ارزهای دیجیتال کمک کند!

چطور کار می‌کند؟
جمع‌آوری داده‌ها: ربات با استفاده از کتابخانه CCXT داده‌های OHLCV (Open, High, Low, Close, Volume) از صرافی KuCoin را برای ارز دیجیتال انتخابی شما دریافت می‌کند.

پیش‌پردازش داده‌ها: داده‌ها با استفاده از MinMaxScaler نرمال می‌شوند و برای آموزش مدل LSTM آماده می‌شوند.

آموزش مدل LSTM: مدل LSTM با داده‌های تاریخی آموزش می‌بیند تا قیمت‌های آینده را پیش‌بینی کند.

پیش‌بینی قیمت: پس از آموزش، ربات از مدل LSTM برای پیش‌بینی قیمت ارز دیجیتال و بازه زمانی مشخص استفاده می‌کند.
