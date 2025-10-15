#!/usr/bin/env python3
"""
ДЗ3: Обучение широкой и глубокой моделей с профилированием
"""

import os
import time
import psutil
import torch
import subprocess
from pathlib import Path

def run_training(model_config, log_dir, model_name):
    """Запускает обучение модели с профилированием"""
    print(f"\n{'='*60}")
    print(f"ЗАПУСК ОБУЧЕНИЯ: {model_name.upper()}")
    print(f"{'='*60}")
    
    # Создаем директорию для логов
    os.makedirs(log_dir, exist_ok=True)
    print(f"📁 Создана директория логов: {log_dir}")
    
    # Запускаем обучение
    start_time = time.time()
    
    try:
        # Запускаем обучение через subprocess для изоляции памяти
        cmd = [
            "python", "train.py",
            f"model={model_config}",
            f"hydra.run.dir={log_dir}",
            "training=hw3",  # используем конфиг с max_steps=300 и batch_size=32
            f"log_dir={os.path.join(log_dir, 'tb')}"  # пишем TensorBoard логи в папку запуска
        ]
        
        print(f"🚀 Команда: {' '.join(cmd)}")
        print(f"⚙️  Конфиг обучения: training=hw3 (max_steps=300, batch_size=32)")
        print("📊 Реалтайм логи будут показаны ниже:")
        print("-" * 60)
        
        # Простой запуск с реалтайм выводом
        result = subprocess.run(cmd)
        
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"\n📊 РЕЗУЛЬТАТЫ ОБУЧЕНИЯ {model_name.upper()}:")
        print(f"⏱️  Время обучения: {training_time:.2f} секунд")
        
        if result.returncode == 0:
            print("✅ Обучение завершено успешно")
        else:
            print(f"❌ Ошибка при обучении (код возврата: {result.returncode})")
            
        return {
            'model_name': model_name,
            'training_time': training_time,
            'success': result.returncode == 0,
            'log_dir': log_dir
        }
        
    except Exception as e:
        print(f"❌ Ошибка при запуске обучения: {e}")
        return {
            'model_name': model_name,
            'training_time': 0,
            'success': False,
            'log_dir': log_dir
        }

def main():
    print("🎯 ДЗ3: Сравнение широкой и глубокой моделей Llama")
    print("=" * 60)
    print(f"⚙️  Гиперпараметры берутся из training=hw3 (max_steps=300, batch_size=32)")
    print("=" * 60)
    
    # Проверяем доступность GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🖥️  Устройство: {device}")
    
    if device == "cuda":
        print(f"🎮 GPU: {torch.cuda.get_device_name()}")
        print(f"💾 GPU память: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Результаты обучения
    results = []
    
    # 1. Широкая модель
    print(f"\n🔄 Начинаем обучение широкой модели...")
    wide_result = run_training(
        model_config="llama_wide",
        log_dir="./logs_hw3/wide_model",
        model_name="Широкая модель"
    )
    results.append(wide_result)
    
    # Очищаем память между запусками
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("🧹 Очищена GPU память")
    
    # 2. Глубокая модель
    print(f"\n🔄 Начинаем обучение глубокой модели...")
    deep_result = run_training(
        model_config="llama_deep", 
        log_dir="./logs_hw3/deep_model",
        model_name="Глубокая модель"
    )
    results.append(deep_result)
    
    # Выводим итоговое сравнение
    print(f"\n{'='*60}")
    print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ")
    print(f"{'='*60}")
    
    for result in results:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['model_name']}:")
        print(f"  Время обучения: {result['training_time']:.2f} сек")
        print(f"  Логи: {os.path.join(result['log_dir'], 'tb')}")
        print()

if __name__ == "__main__":
    main()