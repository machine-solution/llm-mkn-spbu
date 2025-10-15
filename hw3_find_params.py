#!/usr/bin/env python3
"""
ДЗ3: Подбор параметров для моделей одинакового размера
"""

from llama import LlamaConfig, LlamaModel

def count_params(config):
    """Подсчитывает параметры модели"""
    model = LlamaModel(config)
    total = sum(p.numel() for p in model.parameters())
    
    # Подсчитаем компоненты
    embedding_params = config.vocab_size * config.hidden_size
    output_params = config.hidden_size * config.vocab_size
    layer_params = (total - embedding_params - output_params) // config.n_layers
    
    print(f"    Embedding: {embedding_params:,}, Output: {output_params:,}, Per layer: {layer_params:,}")
    
    return total

def find_equal_models():
    """Находит модели примерно одинакового размера"""
    
    # Пересчитаем целевые параметры для уменьшенного vocab_size
    # Базовая модель: n_layers=1, n_heads=2, hidden_size=32, vocab_size=4096
    base_config = LlamaConfig(n_layers=1, n_heads=2, vocab_size=4096, hidden_size=32, seq_len=32)
    target_params = count_params(base_config)
    tolerance = 200000  # Допуск ±200k
    
    print(f"ДЗ3: Подбор слоев для разных hidden_size")
    print(f"Целевые параметры: ~{target_params:,}")
    print(f"Допуск: ±{tolerance:,}")
    print("=" * 60)
    
    # Для каждого hidden_size (степени двойки от 16 до 64) подберем количество слоев
    hidden_sizes = [16, 32, 64]  # 2^4, 2^5, 2^6
    
    results = []
    
    for hidden_size in hidden_sizes:
        print(f"\nhidden_size = {hidden_size}:")
        
        # Пробуем разное количество слоев для данного hidden_size
        # Фиксируем n_heads=2 (как в базовой модели)
        for n_layers in range(1, 50):  # До 49 слоев (больше для hidden_size=64)
            try:
                config = LlamaConfig(
                    n_layers=n_layers,
                    n_heads=2,  # Фиксируем количество голов
                    vocab_size=4096,  # Уменьшенный vocab для ДЗ3
                    hidden_size=hidden_size,
                    seq_len=32
                )
                
                params = count_params(config)
                diff = abs(params - target_params)
                
                if diff <= tolerance:
                    print(f"  {n_layers:2d} слоев -> {params:8,} (diff: {diff:6,})")
                    results.append((config, params, diff, f"hidden={hidden_size}, layers={n_layers}"))
                
            except Exception as e:
                continue
    
    # Найдем лучшие варианты
    print(f"\n{'='*60}")
    print(f"ИТОГОВЫЕ РЕЗУЛЬТАТЫ (diff <= {tolerance:,}):")
    print(f"{'='*60}")
    
    if results:
        # Сортируем по разнице от целевого значения
        results.sort(key=lambda x: x[2])
        
        print(f"Найдено {len(results)} подходящих конфигураций:")
        print()
        
        for i, (config, params, diff, desc) in enumerate(results[:10]):  # Показываем топ-10
            print(f"{i+1:2d}. {desc:20s} -> {params:8,} параметров (diff: {diff:6,})")
        
        # Найдем лучшую пару: одна широкая (больший hidden_size), одна глубокая (больше слоев)
        wide_models = [r for r in results if r[0].hidden_size > 32]  # Широкие модели
        deep_models = [r for r in results if r[0].hidden_size == 32 and r[0].n_layers > 1]  # Глубокие модели
        
        print(f"\nШирокие модели (hidden_size > 32):")
        for config, params, diff, desc in wide_models:
            print(f"  {desc} -> {params:,} параметров")
        
        print(f"\nГлубокие модели (hidden_size=32, layers > 1):")
        for config, params, diff, desc in deep_models:
            print(f"  {desc} -> {params:,} параметров")
        
        if wide_models and deep_models:
            # Выберем лучшую широкую и лучшую глубокую
            best_wide = min(wide_models, key=lambda x: x[2])  # Минимальная разница от целевого
            best_deep = min(deep_models, key=lambda x: x[2])
            
            print(f"\nЛУЧШАЯ ПАРА ДЛЯ СРАВНЕНИЯ:")
            print(f"Широкая модель: {best_wide[3]} -> {best_wide[1]:,} параметров")
            print(f"Глубокая модель: {best_deep[3]} -> {best_deep[1]:,} параметров")
            
            diff_between = abs(best_wide[1] - best_deep[1])
            print(f"Разница между моделями: {diff_between:,} параметров")
            
            print(f"\nКОНФИГУРАЦИИ ДЛЯ ФАЙЛОВ:")
            print(f"\nШирокая модель (conf/model/llama_wide.yaml):")
            print(f"  n_layers: {best_wide[0].n_layers}")
            print(f"  n_heads: {best_wide[0].n_heads}")
            print(f"  hidden_size: {best_wide[0].hidden_size}")
            
            print(f"\nГлубокая модель (conf/model/llama_deep.yaml):")
            print(f"  n_layers: {best_deep[0].n_layers}")
            print(f"  n_heads: {best_deep[0].n_heads}")
            print(f"  hidden_size: {best_deep[0].hidden_size}")
            
            return best_wide[0], best_deep[0]
        else:
            print(f"\n❌ Не найдено подходящих пар (нужны и широкие, и глубокие модели)")
            return None, None
    
    else:
        print("❌ Не найдено подходящих моделей")
        return None, None

if __name__ == "__main__":
    find_equal_models()
