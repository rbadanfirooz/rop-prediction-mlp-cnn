"""
پروژه: پیش‌بینی نرخ حفاری (Rate of Penetration - ROP)
فایل: /mnt/data/MudLogging_Data.xlsx

توضیحات کلی:
این اسکریپت پایتون دو مدل عصبی را برای پیش‌بینی ROP پیاده‌سازی، آموزش و مقایسه می‌کند:
 1) شبکه عصبی چندلایه (MLP)
 2) شبکه عصبی کانولوشنی یک‌بعدی (1D-CNN)
"""

# ================================================================
# بخش اول: وارد کردن کتابخانه‌ها (Imports)
# ================================================================

import os          # برای کار با متغیرهای محیطی سیستم‌عامل
import random      # برای تولید اعداد تصادفی و تنظیم seed
import numpy as np # کتابخانه اصلی محاسبات عددی آرایه‌ای (نام مستعار np)
import pandas as pd # کتابخانه پردازش داده جدولی (نام مستعار pd)
import matplotlib.pyplot as plt # کتابخانه رسم نمودار

# توابع مورد نیاز از scikit-learn برای ML کلاسیک
from sklearn.model_selection import train_test_split  # تقسیم داده به train/test
from sklearn.preprocessing import StandardScaler      # استانداردسازی ویژگی‌ها (میانگین=0، واریانس=1)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score  # معیارهای ارزیابی

# کتابخانه TensorFlow و زیرماژول‌های Keras برای ساخت شبکه عصبی
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
# layers   → لایه‌های شبکه (Dense, Conv1D, ...)
# models   → کلاس‌های Sequential و Model برای ساخت مدل
# callbacks → توابع کمکی در حین آموزش (EarlyStopping, ModelCheckpoint)
# optimizers → بهینه‌سازها مثل Adam


# ================================================================
# بخش دوم: تنظیم Seed برای تولیدمثل‌پذیری نتایج
# ================================================================

RANDOM_SEED = 42  # عدد دلخواه به عنوان seed (هر عدد صحیح مناسب است)

# تنظیم seed در تمام کتابخانه‌ها تا اجراهای مختلف نتیجه یکسان بدهند:
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)  # seed هش‌سازی داخلی پایتون
random.seed(RANDOM_SEED)    # seed کتابخانه random پایتون
np.random.seed(RANDOM_SEED) # seed کتابخانه NumPy
tf.random.set_seed(RANDOM_SEED) # seed TensorFlow (شامل Keras)


# ================================================================
# بخش سوم: پارامترهای قابل تنظیم (Hyperparameters & Config)
# ================================================================

EXCEL_PATH = 'MudLogging_Data.xlsx'  # مسیر فایل اکسل ورودی
TARGET_COLUMN = 'ROP m/hr'           # نام ستونی که باید پیش‌بینی شود (متغیر هدف)
DROP_COLUMNS = ['PUMP 3','RESERVE 1','Vol+/-']  # ستون‌هایی که باید از داده حذف شوند
TEST_SIZE = 0.15   # 15% از کل داده برای تست کنار گذاشته می‌شود
VALID_SIZE = 0.15  # 15% از کل داده برای اعتبارسنجی در نظر گرفته می‌شود
BATCH_SIZE = 32    # تعداد نمونه‌هایی که در هر گام (step) به مدل داده می‌شود
EPOCHS = 200       # حداکثر تعداد دوره‌های آموزش
PATIENCE = 12      # اگر val_loss طی 12 epoch بهتر نشد، آموزش متوقف شود (EarlyStopping)
LEARNING_RATE = 1e-3  # نرخ یادگیری اولیه برای بهینه‌ساز Adam (= 0.001)


# ================================================================
# بخش چهارم: توابع اصلی
# ================================================================

def find_target_column(df):
    """
    این تابع نام ستون هدف (ROP) را در DataFrame پیدا می‌کند.
    اولویت: 1) بررسی TARGET_COLUMN، 2) جستجوی خودکار، 3) انتخاب ستون آخر
    """
    
    # اگر TARGET_COLUMN مقداری دارد، ابتدا همان را جستجو می‌کنیم
    if TARGET_COLUMN:
        if TARGET_COLUMN in df.columns:
            # ستون مورد نظر در DataFrame وجود دارد، مستقیم برمی‌گردانیم
            return TARGET_COLUMN
        else:
            # ستون پیدا نشد، اخطار می‌دهیم و به مرحله بعد می‌رویم
            print(f"هشدار: ستون مورد نظر '{TARGET_COLUMN}' در فایل پیدا نشد. تلاش برای پیدا کردن خودکار...")
    
    # جستجوی خودکار: ستون‌هایی که نامشان شامل کلمات مرتبط با ROP است
    candidates = [c for c in df.columns if any(key in c.lower() for key in ['rop', 'rate', 'rate of penetration', 'penetration'])]
    # توضیح: c.lower() نام ستون را به حروف کوچک تبدیل می‌کند تا جستجو case-insensitive باشد
    
    if candidates:
        # اولین گزینه مناسب پیدا شد
        print(f"ستون هدف تشخیص داده شد: {candidates[0]}")
        return candidates[0]
    
    # اگر هیچ گزینه‌ای پیدا نشد، ستون آخر را به عنوان هدف برمی‌گردانیم
    print("هشدار: ستون هدف با نام شناخته‌شده پیدا نشد. ستون آخر به‌عنوان هدف انتخاب می‌شود:", df.columns[-1])
    return df.columns[-1]  # df.columns[-1] → آخرین ستون


def load_and_preprocess(path):
    """
    بارگذاری فایل اکسل و انجام پیش‌پردازش کامل:
    خروجی: X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler
    """
    
    # ---- گام ۱: بارگذاری فایل اکسل ----
    df = pd.read_excel(path, sheet_name=0)
    # sheet_name=0 → اولین شیت فایل خوانده می‌شود
    print('شکل اولیه داده‌ها:', df.shape)  # (تعداد ردیف، تعداد ستون) چاپ می‌شود

    # ---- گام ۲: حذف ردیف‌های واحد مثل [m/h] ----
    # بعضی فایل‌های لاگ‌حفاری یک ردیف توضیحی مثل [m/h] دارند که داده نیست
    df = df[~df.astype(str).apply(lambda row: row.str.contains(r'\[.*\]').any(), axis=1)]
    # توضیح گام به گام:
    # df.astype(str)                         → همه مقادیر به رشته تبدیل می‌شوند
    # .apply(lambda row: ..., axis=1)        → روی هر ردیف تابع اعمال می‌شود
    # row.str.contains(r'\[.*\]')            → بررسی می‌کند آیا مقداری بین [ ] وجود دارد
    # .any()                                 → اگر حتی یک ستون چنین مقداری داشت True برمی‌گردد
    # ~ (tilde)                              → عملگر NOT، ردیف‌هایی که واحد ندارند نگه می‌دارد

    # ---- گام ۳: پاک‌سازی واحدها از داده‌های عددی ----
    for col in df.columns:
        # هر ستون را به رشته تبدیل و با regex تمام کاراکترهای غیر از عدد، نقطه و منفی را حذف می‌کند
        df[col] = df[col].astype(str).str.replace(r'[^\d\.\-]', '', regex=True)
        # [^\d\.\-] → هر چیزی که رقم (d)، نقطه (.) یا منفی (-) نباشد

    df = df.reset_index(drop=True)  # ایندکس‌های ردیف را از صفر مرتب می‌کند

    # ---- گام ۴: حذف ستون‌های مشخص‌شده توسط کاربر ----
    for c in DROP_COLUMNS:
        if c in df.columns:
            df.drop(columns=c, inplace=True)
            # inplace=True → تغییر مستقیم در df بدون نیاز به انتساب مجدد

    # ---- گام ۵: جداسازی متغیر هدف (y) از ویژگی‌ها (X) ----
    target_col = find_target_column(df)  # پیدا کردن نام ستون هدف
    
    y = pd.to_numeric(df[target_col], errors='coerce')
    # pd.to_numeric → مقادیر را به عدد تبدیل می‌کند
    # errors='coerce' → اگر تبدیل ممکن نبود، NaN قرار می‌دهد (به جای خطا)

    X = df.drop(columns=[target_col])       # همه ستون‌ها به جز هدف
    X = X.apply(pd.to_numeric, errors='coerce')  # تبدیل همه ویژگی‌ها به عدد

    # ---- گام ۶: حذف ردیف‌هایی که مقدار هدف ندارند ----
    mask = y.notna()   # mask یک آرایه Boolean است؛ True = مقدار دارد، False = NaN
    X = X.loc[mask]    # فقط ردیف‌هایی که y آنها مقدار دارد نگه می‌داریم
    y = y.loc[mask]    # متناظراً y را هم فیلتر می‌کنیم

    # ---- گام ۷: حذف ستون‌هایی که کاملاً خالی (NaN) هستند ----
    X = X.dropna(axis=1, how='all')
    # axis=1 → روی ستون‌ها عمل می‌کند
    # how='all' → فقط ستونی حذف شود که تمام مقادیرش NaN باشد

    print("Remaining columns:", X.columns.tolist())  # ستون‌های باقی‌مانده چاپ می‌شود

    # ---- گام ۸: پر کردن مقادیر گم‌شده (NaN) با میانه ستون ----
    X = X.fillna(X.median())  # هر NaN با میانه همان ستون جایگزین می‌شود (robust to outliers)
    y = y.fillna(y.median())  # اگر هنوز NaN در y باشد، با میانه پر می‌شود

    # ---- گام ۹: تقسیم داده به train / validation / test ----
    
    # ابتدا 15% داده برای تست جدا می‌شود
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,      # 15% تست
        random_state=RANDOM_SEED  # برای تکرارپذیری
    )
    
    # محاسبه نسبت validation از داده باقی‌مانده (temp)
    # اگر کل=100%، test=15%، بقیه=85%
    # می‌خواهیم 15% از کل = val_frac_of_temp از 85% باشد
    # پس val_frac_of_temp = 0.15 / (1 - 0.15) = 0.15 / 0.85 ≈ 0.176
    val_frac_of_temp = VALID_SIZE / (1.0 - TEST_SIZE)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_frac_of_temp,
        random_state=RANDOM_SEED
    )

    print('شکل مجموعه‌ها:')
    print('X_train:', X_train.shape, 'X_val:', X_val.shape, 'X_test:', X_test.shape)
    print("X shape after cleaning:", X.shape)
    print("y shape after cleaning:", y.shape)

    # ---- گام ۱۰: استانداردسازی ویژگی‌ها ----
    scaler = StandardScaler()
    # StandardScaler: هر ویژگی را به گونه‌ای تبدیل می‌کند که میانگین=0 و انحراف معیار=1 شود
    # فرمول: z = (x - mean) / std
    
    X_train_scaled = scaler.fit_transform(X_train)
    # fit_transform: ابتدا میانگین و std را از داده آموزش یاد می‌گیرد (fit)
    # سپس تبدیل را اعمال می‌کند (transform)
    
    X_val_scaled = scaler.transform(X_val)
    # transform فقط: از میانگین و std داده آموزش استفاده می‌کند (مهم! نه از val)
    
    X_test_scaled = scaler.transform(X_test)
    # همینطور برای تست: فقط transform با مقادیر یاد گرفته از train
    # دلیل: در واقعیت داده جدید را نمی‌توان refitکرد

    feature_names = X.columns.tolist()  # لیست نام ویژگی‌ها برای استفاده بعدی

    # بازگشت همه خروجی‌های مورد نیاز
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train.values, y_val.values, y_test.values, feature_names, scaler
    # .values → تبدیل pandas Series به آرایه NumPy


# ================================================================
# بخش پنجم: تعریف معماری مدل‌ها
# ================================================================

def build_mlp(input_dim, lr=LEARNING_RATE):
    """
    ساخت مدل MLP (Multi-Layer Perceptron) برای رگرسیون
    input_dim: تعداد ویژگی‌های ورودی
    lr: نرخ یادگیری
    """
    
    model = models.Sequential([  # Sequential = لایه‌ها به ترتیب پشت سر هم
        
        layers.Input(shape=(input_dim,)),
        # لایه ورودی: بردار به اندازه input_dim ویژگی دریافت می‌کند
        
        layers.Dense(128, activation='relu'),
        # لایه متصل کامل با 128 نورون و تابع فعال‌سازی ReLU
        # ReLU: f(x) = max(0, x) → عدد منفی را صفر می‌کند
        
        layers.Dropout(0.2),
        # Dropout: در هر گام آموزش، 20% از نورون‌ها به صورت تصادفی خاموش می‌شوند
        # هدف: جلوگیری از overfitting (حفظ کردن بیش از حد داده آموزش)
        
        layers.Dense(64, activation='relu'),
        # لایه متصل کامل با 64 نورون
        
        layers.Dropout(0.15),
        # Dropout با نرخ 15%
        
        layers.Dense(32, activation='relu'),
        # لایه متصل کامل با 32 نورون (بدون Dropout)
        
        layers.Dense(1, activation='linear')
        # لایه خروجی: یک نورون با تابع خطی
        # activation='linear' → مناسب برای رگرسیون (پیش‌بینی عدد پیوسته)
    ])
    
    opt = optimizers.Adam(learning_rate=lr)
    # Adam: یکی از بهترین بهینه‌سازها، ترکیب momentum و adaptive learning rate
    
    model.compile(
        optimizer=opt,    # بهینه‌ساز برای به‌روزرسانی وزن‌ها
        loss='mse',       # تابع خطا: Mean Squared Error (مناسب رگرسیون)
        metrics=['mae']   # معیار نمایشی: Mean Absolute Error
    )
    return model


def build_cnn1d(input_dim, lr=LEARNING_RATE):
    """
    ساخت مدل CNN یک‌بعدی (1D Convolutional Neural Network)
    برای داده جدولی، ویژگی‌ها به شکل دنباله در نظر گرفته می‌شوند.
    input_dim: تعداد ویژگی‌ها
    """
    
    inputs = layers.Input(shape=(input_dim, 1))
    # ورودی: (تعداد ویژگی‌ها، 1 کانال)
    # CNN نیاز دارد داده به شکل (طول دنباله، تعداد کانال) باشد
    
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    # Conv1D: فیلترهای کانولوشن روی دنباله اعمال می‌شوند
    # filters=64   → 64 فیلتر مختلف (هر فیلتر یک الگو یاد می‌گیرد)
    # kernel_size=3 → هر فیلتر به 3 موقعیت متوالی نگاه می‌کند
    # padding='same' → خروجی همان طول ورودی را حفظ می‌کند
    
    x = layers.MaxPooling1D(pool_size=2)(x)
    # MaxPooling: طول دنباله را نصف می‌کند، مقدار بیشینه هر پنجره 2تایی را نگه می‌دارد
    # هدف: کاهش ابعاد و استخراج مهم‌ترین ویژگی‌ها
    
    x = layers.Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
    # لایه کانولوشنی دوم با 32 فیلتر (کمتر از لایه اول → انتزاع بیشتر)
    
    x = layers.GlobalAveragePooling1D()(x)
    # میانگین‌گیری سراسری: خروجی Conv1D را به یک بردار ثابت تبدیل می‌کند
    # (بدون توجه به طول دنباله، یک بردار 32‌تایی تولید می‌شود)
    
    x = layers.Dense(32, activation='relu')(x)
    # لایه Dense بعد از CNN برای ترکیب ویژگی‌های استخراج‌شده
    
    outputs = layers.Dense(1, activation='linear')(x)
    # لایه خروجی رگرسیون: یک مقدار پیوسته پیش‌بینی می‌کند
    
    model = models.Model(inputs=inputs, outputs=outputs)
    # ساخت مدل با API تابعی (Functional API) به جای Sequential
    # مناسب برای معماری‌های پیچیده‌تر
    
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])
    return model


# ================================================================
# بخش ششم: آموزش و ارزیابی مدل
# ================================================================

def train_and_evaluate(model, X_train, y_train, X_val, y_val, X_test, y_test, model_name='model'):
    """
    آموزش مدل و ارزیابی عملکرد روی داده تست
    """
    
    # ---- تعریف Callbacks ----
    
    es = callbacks.EarlyStopping(
        monitor='val_loss',         # معیار نظارت: خطای validation
        patience=PATIENCE,          # اگر 12 epoch بهبودی نبود، متوقف شو
        restore_best_weights=True,  # وزن‌های بهترین epoch را بازیابی کن
        verbose=1                   # پیام توقف را نمایش بده
    )
    # EarlyStopping: از آموزش بیش از حد (overfitting) جلوگیری می‌کند
    
    checkpoint_path = f'{model_name}_best.h5'  # مسیر ذخیره بهترین مدل
    mc = callbacks.ModelCheckpoint(
        checkpoint_path,        # مسیر فایل ذخیره
        monitor='val_loss',     # بر اساس بهترین val_loss ذخیره کن
        save_best_only=True,    # فقط بهترین مدل را ذخیره کن (نه آخری)
        verbose=1
    )
    # ModelCheckpoint: در هر epoch اگر val_loss بهتر شد، مدل ذخیره می‌شود

    # ---- آموزش مدل ----
    history = model.fit(
        X_train, y_train,                   # داده آموزش
        validation_data=(X_val, y_val),     # داده اعتبارسنجی برای نظارت
        epochs=EPOCHS,                      # حداکثر 200 epoch
        batch_size=BATCH_SIZE,              # 32 نمونه در هر گام
        callbacks=[es, mc],                 # callbacks فعال
        verbose=2                           # یک خط خلاصه برای هر epoch
    )
    # model.fit خروجی history برمی‌گرداند که شامل مقادیر loss/mae در هر epoch است

    # ---- بارگذاری بهترین وزن‌های ذخیره‌شده ----
    model.load_weights(checkpoint_path)
    # از آنجا که EarlyStopping ممکن است آموزش را زودتر متوقف کرده باشد
    # بهترین وزن‌ها را از فایل ذخیره‌شده بازیابی می‌کنیم

    # ---- پیش‌بینی و محاسبه معیارهای ارزیابی ----
    y_pred = model.predict(X_test).squeeze()
    # model.predict → پیش‌بینی روی داده تست
    # .squeeze() → حذف ابعاد اضافی (از شکل (n,1) به شکل (n,))

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # RMSE: ریشه میانگین مربعات خطا (واحد = واحد متغیر هدف)
    # فرمول: sqrt(mean((y_true - y_pred)^2))
    
    mae = mean_absolute_error(y_test, y_pred)
    # MAE: میانگین قدر مطلق خطا
    # فرمول: mean(|y_true - y_pred|)
    
    r2 = r2_score(y_test, y_pred)
    # R2 (ضریب تعیین): نشان می‌دهد مدل چه درصدی از واریانس داده را توضیح می‌دهد
    # R2=1 → پیش‌بینی کامل | R2=0 → مدل بهتر از میانگین نیست | R2<0 → بد

    results = {'rmse': rmse, 'mae': mae, 'r2': r2, 'y_pred': y_pred}
    # نتایج در قالب دیکشنری بازگردانده می‌شوند
    
    return model, history, results


# ================================================================
# بخش هفتم: توابع رسم نمودار
# ================================================================

def plot_history(history, title='Training History'):
    """
    رسم نمودار Loss و MAE در طول آموزش
    """
    
    plt.figure(figsize=(10, 4))  # اندازه کل تصویر: 10 اینچ عرض، 4 اینچ ارتفاع
    
    # --- نمودار چپ: Loss ---
    plt.subplot(1, 2, 1)
    # subplot(تعداد ردیف, تعداد ستون, شماره فعلی) → grid 1x2، نمودار اول
    
    plt.plot(history.history['loss'], label='train_loss')
    # history.history['loss'] → لیست مقادیر loss آموزش در هر epoch
    
    plt.plot(history.history['val_loss'], label='val_loss')
    # history.history['val_loss'] → مقادیر loss اعتبارسنجی
    
    plt.title(title + ' - Loss')  # عنوان نمودار
    plt.legend()  # نمایش راهنما (لجند)

    # --- نمودار راست: MAE ---
    plt.subplot(1, 2, 2)  # نمودار دوم در grid
    
    plt.plot(history.history['mae'], label='train_mae')
    plt.plot(history.history['val_mae'], label='val_mae')
    plt.title(title + ' - MAE')
    plt.legend()
    
    plt.tight_layout()  # فاصله‌گذاری خودکار بین نمودارها
    plt.show()          # نمایش نمودار


def plot_pred_vs_true(y_test, y_pred, title='Predicted vs True'):
    """
    رسم نمودار پراکندگی مقادیر پیش‌بینی‌شده در برابر مقادیر واقعی
    """
    
    plt.figure(figsize=(6, 6))  # نمودار مربعی 6x6
    
    plt.scatter(y_test, y_pred, alpha=0.6)
    # scatter plot: محور X = مقادیر واقعی، محور Y = مقادیر پیش‌بینی
    # alpha=0.6 → شفافیت 60% (برای نقاط تداخل‌دار بهتر دیده می‌شود)
    
    mi = min(np.min(y_test), np.min(y_pred))  # کمترین مقدار برای رسم خط مرجع
    ma = max(np.max(y_test), np.max(y_pred))  # بیشترین مقدار
    
    plt.plot([mi, ma], [mi, ma], 'k--')
    # رسم خط قطری با خط چین مشکی (k=black, --=dashed)
    # این خط نشان می‌دهد پیش‌بینی کامل کجاست (y_pred = y_true)
    # هر چه نقاط به این خط نزدیک‌تر باشند، مدل بهتر است
    
    plt.xlabel('True')      # عنوان محور X
    plt.ylabel('Predicted') # عنوان محور Y
    plt.title(title)
    plt.show()


# ================================================================
# بخش هشتم: تابع اصلی (main) - اجرای کل pipeline
# ================================================================

def main():
    
    # ---- مرحله ۱: بارگذاری و پیش‌پردازش داده ----
    print('\n=== بارگذاری و پیش‌پردازش داده ===')
    X_train, X_val, X_test, y_train, y_val, y_test, feature_names, scaler = load_and_preprocess(EXCEL_PATH)
    # scaler ذخیره می‌شود تا بعداً بتوان داده‌های جدید را با همان پارامترها نرمال کرد

    input_dim = X_train.shape[1]
    # X_train.shape = (تعداد نمونه، تعداد ویژگی)
    # shape[1] → تعداد ویژگی‌ها (input_dim) برای تعریف لایه ورودی

    # ---- مرحله ۲: آموزش مدل MLP ----
    print('\n=== آموزش MLP ===')
    mlp = build_mlp(input_dim)  # ساخت مدل
    mlp_model, mlp_history, mlp_results = train_and_evaluate(
        mlp, X_train, y_train, X_val, y_val, X_test, y_test, model_name='mlp'
    )
    print('نتایج MLP:', mlp_results)
    
    plot_history(mlp_history, title='MLP Training')
    # نمودار loss/mae در طول آموزش
    
    plot_pred_vs_true(y_test, mlp_results['y_pred'], title='MLP: Predicted vs True')
    # نمودار پیش‌بینی در برابر واقعی

    # ---- مرحله ۳: آماده‌سازی داده برای CNN1D ----
    print('\n=== آماده‌سازی ورودی برای CNN1D ===')
    
    X_train_c = X_train.reshape((-1, input_dim, 1))
    # reshape: تبدیل شکل آرایه از (n_samples, n_features) به (n_samples, n_features, 1)
    # -1 → پایتون تعداد نمونه‌ها را خودکار محاسبه می‌کند
    # این ری‌شیپ ضروری است چون Conv1D انتظار ورودی 3 بعدی دارد
    
    X_val_c = X_val.reshape((-1, input_dim, 1))    # همان کار برای validation
    X_test_c = X_test.reshape((-1, input_dim, 1))  # همان کار برای test

    # ---- مرحله ۴: آموزش مدل CNN1D ----
    print('\n=== آموزش CNN1D ===')
    cnn = build_cnn1d(input_dim)  # ساخت مدل CNN
    cnn_model, cnn_history, cnn_results = train_and_evaluate(
        cnn, X_train_c, y_train, X_val_c, y_val, X_test_c, y_test, model_name='cnn1d'
    )
    print('نتایج CNN1D:', cnn_results)
    
    plot_history(cnn_history, title='CNN1D Training')
    plot_pred_vs_true(y_test, cnn_results['y_pred'], title='CNN1D: Predicted vs True')

    # ---- مرحله ۵: مقایسه نهایی دو مدل ----
    print('\n=== مقایسه مدل‌ها ===')
    summary = pd.DataFrame({
        'model': ['MLP', 'CNN1D'],              # نام مدل‌ها
        'rmse': [mlp_results['rmse'], cnn_results['rmse']],  # مقادیر RMSE
        'mae': [mlp_results['mae'], cnn_results['mae']],     # مقادیر MAE
        'r2': [mlp_results['r2'], cnn_results['r2']]         # مقادیر R2
    })
    # pd.DataFrame → ساخت جدول مقایسه‌ای از دیکشنری
    print(summary)  # چاپ جدول مقایسه

    # ---- مرحله ۶: ذخیره مدل‌ها و scaler ----
    try:
        import joblib  # کتابخانه ذخیره‌سازی آبجکت‌های پایتون
        
        joblib.dump(scaler, 'scaler.save')
        # ذخیره scaler: برای نرمال‌سازی داده‌های جدید در آینده باید از همین scaler استفاده شود
        
        mlp_model.save('mlp_model.h5')
        # ذخیره مدل MLP در فرمت HDF5 (شامل معماری + وزن‌ها)
        
        cnn_model.save('cnn1d_model.h5')
        # ذخیره مدل CNN1D
        
        print('مدل‌ها و scaler ذخیره شدند: mlp_model.h5 , cnn1d_model.h5 , scaler.save')
    except Exception as e:
        # اگر ذخیره‌سازی با خطا مواجه شد، برنامه کرش نمی‌کند
        print('هشدار: ذخیره‌سازی موفق نبود:', e)


# ================================================================
# نقطه ورود برنامه
# ================================================================

if __name__ == '__main__':
    # این بلاک فقط زمانی اجرا می‌شود که فایل مستقیم اجرا شود
    # (نه زمانی که به عنوان ماژول import شود)
    main()