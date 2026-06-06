# Surveillance System

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nour769/surveillance-system/blob/main/demo.ipynb)

> Clique sur le bouton pour tester le projet directement dans ton navigateur - aucune installation requise !

 - Real-Time Intelligent Monitoring

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-Compose-blue?logo=docker)
![MLflow](https://img.shields.io/badge/MLflow-2.11-orange)
![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub_Actions-black?logo=github)

> Systeme de surveillance intelligente en temps reel

---

## Benchmark des modeles

| Modele | Taille | FPS | Detections/frame | Statut |
|--------|--------|-----|-----------------|--------|
| YOLOv8n | 6 MB | 52.4 | 8.6 | Rapide |
| YOLOv8s | 22 MB | 52.8 | 9.9 | RETENU |
| YOLOv8m | 50 MB | 42.0 | 9.3 | Precis |

---

## Lancer le projet

### Avec Docker
```bash
git clone https://github.com/nour769/surveillance-system.git
cd surveillance-system
docker-compose up
```

### Sans Docker
```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

---

## API Endpoints

| Endpoint | Methode | Description |
|----------|---------|-------------|
| /health | GET | Health check |
| /analyze | POST | Analyser une video |
| /results | GET | Tous les resultats |
| /download/{job_id} | GET | Video annotee |

---

## Exemple reponse API

```json
{
  "job_id": "6db09dba",
  "frames_traitees": 460,
  "personnes_uniques": 81,
  "entrees": 3,
  "sorties": 2,
  "fps_traitement": 19.6
}
```

---

## Conformite RGPD
- Floutage automatique des visages
- Aucune donnee biometrique stockee
- IDs anonymes uniquement

---

## Tests
```bash
pytest tests/ -v
# 4 passed in 3.73s
```

---

## Stack technique

| Technologie | Usage |
|-------------|-------|
| YOLOv8s | Detection objets temps reel |
| ByteTrack | Tracking multi-objets |
| FastAPI | API REST |
| Docker + Compose | Containerisation |
| MLflow | Tracking MLOps |
| GitHub Actions | CI/CD |

---

## Auteur
**Nour Mrabet** - [@nour769](https://github.com/nour769)
