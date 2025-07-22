# ΔΣ::TORUS - Управление HF Space

## 🏗️ **Архитектура репозиториев**

### Основной репозиторий (GitHub)
- **URL:** https://github.com/Personaz1/TORUS
- **Назначение:** Основной код проекта, документация, исследования
- **Структура:** `toroidal_diffusion_complete_website/`

### HF Space репозиторий (Hugging Face)
- **URL:** https://huggingface.co/spaces/stephansolncev/TORUS
- **Назначение:** Демо-интерфейс для Hugging Face Spaces
- **Локальная папка:** `TORUS_HF_SPACE/`

## 🔧 **Управление HF Space**

### Обновление HF Space
```bash
cd TORUS_HF_SPACE
git add .
git commit -m "Update HF Space"
git push hf-space main
```

### Клонирование HF Space
```bash
git clone git@hf.co:spaces/stephansolncev/TORUS TORUS_HF_SPACE
cd TORUS_HF_SPACE
```

### Проверка статуса
```bash
cd TORUS_HF_SPACE
git status
git remote -v
```

## 📁 **Структура HF Space**

```
TORUS_HF_SPACE/
├── app.py                    # Основной Gradio интерфейс
├── app_working.py           # Рабочая версия (schema-safe)
├── requirements.txt         # Зависимости
├── README.md               # Документация
├── torusq_quantum_core.py  # Квантовое ядро
├── torusq_quantum_interface.py # Интерфейс сознания
└── UPDATE_HF_SPACE.md      # Инструкции по обновлению
```

## ⚠️ **Важные правила**

1. **НЕ добавлять TORUS_HF_SPACE в основной репозиторий**
2. **Использовать SSH для HF Space:** `git@hf.co:spaces/stephansolncev/TORUS`
3. **Обновлять HF Space отдельно от основного проекта**
4. **Тестировать schema-совместимость перед деплоем**

## 🚀 **Быстрое обновление**

```bash
# Обновить HF Space
cd TORUS_HF_SPACE
git add .
git commit -m "Fix schema compatibility"
git push hf-space main

# Проверить статус
curl -s https://huggingface.co/spaces/stephansolncev/TORUS
```

## 🔍 **Отладка**

### Schema ошибки
- Убрать `precision=` из `gr.Number`
- Убрать `variant=` из `gr.Button`
- Убрать `size=` из кнопок
- Обновить Gradio до >=4.50.0

### Проблемы с доступом
- Использовать SSH ключи для HF
- Проверить права доступа к Space
- Убедиться в правильности remote URL 