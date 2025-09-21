# Llama Implementation

Реализация архитектуры Llama с нуля на PyTorch.

## Компоненты

- **LlamaConfig** - конфигурация модели
- **SelfAttention** - multi-head self-attention с causal mask
- **RoPE** - ротационное позиционное кодирование
- **RMSNorm** - нормализация слоев
- **SwiGLU** - Feed Forward Network с активацией SiLU
- **LlamaBlock** - полный блок трансформера
- **LlamaModel** - полная модель с генерацией текста

## Использование

```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск тестов
python main.py
```

## Тестирование

Скрипт `main.py` проверяет:
- Forward pass с рандомными токенами
- Backward pass с вычислением градиентов
- Корректность размеров выходов
