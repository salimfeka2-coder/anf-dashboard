# main.py - Application d'Analyse ANF Complète avec Module Conclusions & Recommandations

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import base64
import json
import os
import io
import re
from difflib import get_close_matches
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import tempfile

# Configuration de la page
st.set_page_config(
    page_title="Analyse des données ANF",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

# Ajouter cette fonction pour éclater les combinaisons de technologies
def eclater_technologies(df):
    """
    Éclate les combinaisons de technologies et compte chaque technologie séparément
    """
    # Créer une copie du DataFrame
    df_eclate = df.copy()
    
    # Fonction pour séparer les technologies combinées
    def separer_technologies(tech):
        if not isinstance(tech, str):
            return [tech]
        
        # Séparer par slash tout en gardant les technologies individuelles
        if '/' in tech:
            return [t.strip() for t in tech.split('/')]
        else:
            return [tech]
    
    # Appliquer la séparation
    df_eclate['technologie'] = df_eclate['technologie'].apply(separer_technologies)
    
    # Éclater les listes en plusieurs lignes
    df_eclate = df_eclate.explode('technologie')
    
    return df_eclate

def normaliser_texte(texte):
    """Normalise un texte pour la comparaison"""
    if not isinstance(texte, str):
        texte = str(texte)
    
    # Convertir en minuscules
    texte = texte.lower()
    
    # Supprimer les accents
    accents = {'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e', 
               'à': 'a', 'â': 'a', 'ä': 'a', 
               'î': 'i', 'ï': 'i', 
               'ô': 'o', 'ö': 'o', 
               'ù': 'u', 'û': 'u', 'ü': 'u', 
               'ÿ': 'y', 'ç': 'c'}
    
    for acc, sans_acc in accents.items():
        texte = texte.replace(acc, sans_acc)
    
    # Supprimer les caractères spéciaux et les espaces
    texte = re.sub(r'[^a-z0-9]', '', texte)
    
    return texte

def mapper_colonnes(df):
    """Map les colonnes du DataFrame aux noms standards"""
    # Création d'une copie
    df = df.copy()
    
    # Dictionnaire des variantes possibles
    mappings = {
        'site_id': ['site_id', 'site id', 'siteid', 'id_site', 'id site', 'identifiant', 'id', 'site', 'numero', 'num'],
        'technologie': ['technologie', 'technologie_anf', 'tech', 'technologie_telecom', 'type_technologie', 'techno', 'technology'],
        'region': ['region', 'région', 'reg', 'nom_region', 'nom région', 'zone', 'area'],
        'wilaya': ['wilaya', 'wilaya_nom', 'nom_wilaya', 'province', 'département', 'departement', 'dept', 'commune'],
        'date_depot_anf': ['date_depot_anf', 'date_depot', 'date depot', 'date', 'depot_date', 'date_anf', 'dateanf', 'date_soumission', 'submission_date'],
        'avis': ['avis', 'avis_anf', 'decision', 'statut', 'résultat', 'resultat', 'approval', 'validation', 'status']
    }
    
    # Normaliser les noms de colonnes
    colonnes_normalisees = {col: normaliser_texte(col) for col in df.columns}
    
    # Mapper les colonnes
    colonnes_mappees = {}
    colonnes_non_mappees = []
    
    for col_std in mappings.keys():
        trouve = False
        for col_orig, col_norm in colonnes_normalisees.items():
            if col_norm in [normaliser_texte(v) for v in mappings[col_std]]:
                colonnes_mappees[col_std] = col_orig
                trouve = True
                break
        
        if not trouve:
            # Essayer une correspondance approximative
            noms_colonnes_norm = list(colonnes_normalisees.values())
            matches = get_close_matches(col_std, noms_colonnes_norm, n=1, cutoff=0.6)
            if matches:
                for col_orig, col_norm in colonnes_normalisees.items():
                    if col_norm == matches[0]:
                        colonnes_mappees[col_std] = col_orig
                        trouve = True
                        break
        
        if not trouve:
            colonnes_non_mappees.append(col_std)
    
    # Renommer les colonnes
    for col_std, col_orig in colonnes_mappees.items():
        df.rename(columns={col_orig: col_std}, inplace=True)
    
    return df, colonnes_mappees, colonnes_non_mappees

def suggerer_corrections_colonnes(colonnes_manquantes, df):
    """Suggère des corrections pour les colonnes manquantes"""
    suggestions = {}
    
    patterns = {
        'site_id': r'(?:id|site|identifiant|numero)',
        'technologie': r'(?:tech|technologie)',
        'region': r'(?:region|zone)',
        'wilaya': r'(?:wilaya|province|dept|commune)',
        'date_depot_anf': r'(?:date|depot)',
        'avis': r'(?:avis|decision|statut|resultat)'
    }
    
    for col_manquante in colonnes_manquantes:
        pattern = patterns.get(col_manquante, r'')
        colonnes_correspondantes = []
        
        for col in df.columns:
            if re.search(pattern, col.lower()):
                colonnes_correspondantes.append(col)
        
        if colonnes_correspondantes:
            suggestions[col_manquante] = colonnes_correspondantes
    
    return suggestions

def pretraiter_donnees(df):
    """Prétraite les données selon vos spécifications"""
    df = df.copy()
    
    # Éclater les technologies combinées avant le traitement
    df = eclater_technologies(df)
    
    # Gestion des dates
    df['date_depot_anf'] = pd.to_datetime(df['date_depot_anf'], errors='coerce')
    
    # Remplir les dates manquantes avec la date médiane
    if df['date_depot_anf'].isna().any():
        date_mediane = df['date_depot_anf'].median()
        if pd.isna(date_mediane):
            df['date_depot_anf'] = df['date_depot_anf'].fillna(pd.to_datetime('today'))
        else:
            df['date_depot_anf'] = df['date_depot_anf'].fillna(date_mediane)
    
    # Créer des colonnes temporelles
    df['annee'] = df['date_depot_anf'].dt.year
    df['trimestre'] = df['date_depot_anf'].dt.quarter
    df['mois'] = df['date_depot_anf'].dt.month
    df['periode'] = df['annee'].astype(str) + "-T" + df['trimestre'].astype(str)
    
    # MAPPING PERSONNALISÉ DES AVIS selon vos spécifications
    if 'avis' in df.columns:
        def mapper_avis_personnalise(avis_val):
            avis_str = str(avis_val).strip().lower()
            
            # Mapping exact selon vos besoins
            if avis_str in ['ok', 'okay']:
                return 'favorable'
            elif avis_str in ['favorable', 'favorble']:  # typo courante
                return 'favorable'
            elif avis_str in ['défavorable', 'defavorable', 'unfavorable']:
                return 'défavorable'
            elif avis_str in ['en instance', 'instance', 'pending']:
                return 'en instance'
            elif avis_str in ['nan', 'null', '', 'none']:
                return 'Non spécifié'
            else:
                # Pour les valeurs numériques comme 13447, on les garde telles quelles
                return str(avis_val)
        
        df['avis'] = df['avis'].apply(mapper_avis_personnalise)
    
    # Extraire les valeurs uniques (ordre spécifique pour les avis)
    avis_ordre = ['favorable', 'défavorable', 'en instance', 'Non spécifié']
    avis_uniques = [x for x in avis_ordre if x in df['avis'].unique()]
    # Ajouter les autres valeurs non mappées
    autres_avis = [x for x in df['avis'].unique() if x not in avis_ordre]
    avis_uniques.extend(sorted(autres_avis))
    
    valeurs_uniques = {
        'technologie': sorted([x for x in df['technologie'].unique() if pd.notna(x)]),
        'region': sorted([x for x in df['region'].unique() if pd.notna(x)]),
        'wilaya': sorted([x for x in df['wilaya'].unique() if pd.notna(x)]),
        'avis': avis_uniques,
        'periode': sorted([x for x in df['periode'].unique() if pd.notna(x)])
    }
    
    # Calculer les métriques
    metriques = {
        'total_anf': len(df),
        'technologies_count': df['technologie'].value_counts().to_dict(),
        'regions_count': df['region'].value_counts().to_dict(),
        'wilayas_count': df['wilaya'].value_counts().to_dict(),
        'avis_count': df['avis'].value_counts().to_dict(),
    }
    
    return {
        'df': df,
        'valeurs_uniques': valeurs_uniques,
        'metriques': metriques
    }

def appliquer_filtres(df, filtres):
    """Applique les filtres sélectionnés au DataFrame"""
    df_filtre = df.copy()
    
    if filtres.get('technologie'):
        df_filtre = df_filtre[df_filtre['technologie'].isin(filtres['technologie'])]
    
    if filtres.get('region'):
        df_filtre = df_filtre[df_filtre['region'].isin(filtres['region'])]
    
    if filtres.get('wilaya'):
        df_filtre = df_filtre[df_filtre['wilaya'].isin(filtres['wilaya'])]
    
    if filtres.get('avis'):
        df_filtre = df_filtre[df_filtre['avis'].isin(filtres['avis'])]
    
    if filtres.get('periode'):
        df_filtre = df_filtre[df_filtre['periode'].isin(filtres['periode'])]
    
    return df_filtre

# =============================================================================
# FONCTIONS DE VISUALISATION
# =============================================================================

def create_kpi_metrics(df_filtre):
    """Crée les métriques KPI basées sur les données filtrées"""
    if df_filtre.empty:
        return {
            'Total ANF': 0,
            'Technologies': 0,
            'Régions': 0,
            'Wilayas': 0,
            '% Favorable': 0
        }
    
    kpis = {
        'Total ANF': len(df_filtre),
        'Technologies': df_filtre['technologie'].nunique(),
        'Régions': df_filtre['region'].nunique(),
        'Wilayas': df_filtre['wilaya'].nunique(),
        '% Favorable': round((df_filtre['avis'] == 'favorable').sum() / len(df_filtre) * 100, 1) if len(df_filtre) > 0 else 0
    }
    
    return kpis

def create_technology_pie_chart(df_filtre):
    """Crée le graphique circulaire des technologies (filtrable)"""
    if df_filtre.empty:
        fig = go.Figure()
        fig.update_layout(title='Aucune donnée disponible avec les filtres actuels')
        return fig
    
    # Compter les technologies après éclatement
    tech_counts = df_filtre['technologie'].value_counts()
    
    df_tech = pd.DataFrame({
        'Technologie': tech_counts.index,
        'Nombre': tech_counts.values
    })
    
    fig = px.pie(
        df_tech, 
        values='Nombre', 
        names='Technologie',
        title='Répartition des ANF par Technologie (éclatée)',
        hole=0.4
    )
    
    fig.update_traces(textposition='inside', textinfo='percent+label')
    fig.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def create_technology_summary_table(df_filtre):
    """Crée un tableau récapitulatif des technologies avec leurs comptes"""
    if df_filtre.empty:
        return None
    
    # Compter les technologies
    tech_counts = df_filtre['technologie'].value_counts().reset_index()
    tech_counts.columns = ['Technologie', 'Nombre']
    
    # Calculer les pourcentages
    total = tech_counts['Nombre'].sum()
    tech_counts['Pourcentage'] = round((tech_counts['Nombre'] / total) * 100, 2)
    
    return tech_counts

def create_top_wilayas_favorable_chart(df_filtre):
    """Crée le graphique des top 5 wilayas par nombre d'avis favorables (filtrable)"""
    if df_filtre.empty:
        fig = go.Figure()
        fig.update_layout(title='Aucune donnée disponible avec les filtres actuels')
        return fig
    
    # Filtrer uniquement les ANF favorables
    df_favorable = df_filtre[df_filtre['avis'] == 'favorable']
    
    if df_favorable.empty:
        fig = go.Figure()
        fig.update_layout(title='Aucun avis favorable trouvé avec les filtres actuels')
        return fig
    
    # Compter les ANF favorables par wilaya
    top_wilayas = df_favorable['wilaya'].value_counts().head(5).reset_index()
    top_wilayas.columns = ['wilaya', 'nombre_anf_favorable']
    
    # Créer le graphique
    fig = px.bar(
        top_wilayas,
        x='nombre_anf_favorable',
        y='wilaya',
        orientation='h',
        title='Top 5 Wilayas par Nombre d\'Avis Favorables',
        color='nombre_anf_favorable',
        color_continuous_scale='Greens',
        text='nombre_anf_favorable'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(yaxis={'categoryorder': 'total ascending'})
    fig.update_xaxes(title='Nombre d\'Avis Favorables')
    fig.update_yaxes(title='Wilaya')
    
    return fig

def create_technology_by_region_chart(df_filtre):
    """Crée le graphique de répartition des technologies par région (filtrable)"""
    if df_filtre.empty:
        fig = go.Figure()
        fig.update_layout(title='Aucune donnée disponible avec les filtres actuels')
        return fig
    
    # Créer le tableau croisé
    tech_region = pd.crosstab(df_filtre['region'], df_filtre['technologie'])
    
    # Convertir en format long pour plotly
    tech_region_long = tech_region.reset_index().melt(
        id_vars='region', 
        var_name='technologie', 
        value_name='nombre'
    )
    
    # Créer le graphique en barres empilées
    fig = px.bar(
        tech_region_long,
        x='region',
        y='nombre',
        color='technologie',
        title='Répartition des Technologies par Région',
        labels={'nombre': 'Nombre d\'ANF', 'region': 'Région'},
        text='nombre'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    
    return fig

def create_technology_by_wilaya_chart(df_filtre):
    """Crée le graphique de répartition des technologies par wilaya (filtrable)"""
    if df_filtre.empty:
        fig = go.Figure()
        fig.update_layout(title='Aucune donnée disponible avec les filtres actuels')
        return fig
    
    # Prendre les top 10 wilayas pour éviter un graphique trop chargé
    top_wilayas = df_filtre['wilaya'].value_counts().head(10).index
    df_top_wilayas = df_filtre[df_filtre['wilaya'].isin(top_wilayas)]
    
    # Créer le tableau croisé
    tech_wilaya = pd.crosstab(df_top_wilayas['wilaya'], df_top_wilayas['technologie'])
    
    # Convertir en format long pour plotly
    tech_wilaya_long = tech_wilaya.reset_index().melt(
        id_vars='wilaya', 
        var_name='technologie', 
        value_name='nombre'
    )
    
    # Créer le graphique en barres empilées
    fig = px.bar(
        tech_wilaya_long,
        x='wilaya',
        y='nombre',
        color='technologie',
        title='Répartition des Technologies par Wilaya (Top 10)',
        labels={'nombre': 'Nombre d\'ANF', 'wilaya': 'Wilaya'},
        text='nombre'
    )
    
    fig.update_traces(texttemplate='%{text}', textposition='inside')
    fig.update_layout(
        xaxis_tickangle=-45,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    
    return fig

def create_evolution_line_chart(df_filtre):
    """Crée le graphique d'évolution temporelle en courbes (filtrable)"""
    if df_filtre.empty:
        fig = go.Figure()
        fig.update_layout(title='Aucune donnée disponible avec les filtres actuels')
        return fig
    
    # Grouper par période et avis
    evolution_data = df_filtre.groupby(['periode', 'avis']).size().unstack(fill_value=0).reset_index()
    
    if evolution_data.empty:
        fig = go.Figure()
        fig.update_layout(title='Aucune donnée d\'évolution disponible')
        return fig
    
    # Trier les périodes
    evolution_data = evolution_data.sort_values('periode')
    
    # Créer le graphique en courbes
    fig = go.Figure()
    
    # Ajouter une courbe pour chaque type d'avis
    avis_columns = [col for col in evolution_data.columns if col != 'periode']
    colors = ['#2E8B57', '#DC143C', '#FF8C00', '#4682B4']  # Vert, Rouge, Orange, Bleu
    
    for i, avis in enumerate(avis_columns):
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=evolution_data['periode'],
            y=evolution_data[avis],
            mode='lines+markers',
            name=avis,
            line=dict(color=color, width=3),
            marker=dict(size=8),
            hovertemplate=f'<b>{avis}</b><br>' +
                         'Période: %{x}<br>' +
                         'Nombre: %{y}<br>' +
                         '<extra></extra>'
        ))
    
    fig.update_layout(
        title='Évolution du Nombre de Demandes par Trimestre et par Avis',
        xaxis_title='Période (Trimestre)',
        yaxis_title='Nombre de Demandes',
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Améliorer l'affichage de l'axe X
    fig.update_xaxes(tickangle=45)
    
    return fig

# =============================================================================
# NOUVELLES FONCTIONS D'ANALYSE POUR CONCLUSIONS & RECOMMANDATIONS
# =============================================================================

def analyser_repartition_technologique(df_filtre):
    """Analyse la répartition technologique avec calculs dynamiques"""
    if df_filtre.empty:
        return {}
    
    tech_counts = df_filtre['technologie'].value_counts()
    total = len(df_filtre)
    
    repartition = {}
    for tech, count in tech_counts.items():
        pourcentage = round((count / total) * 100, 1)
        repartition[tech] = {
            'count': count,
            'pourcentage': pourcentage
        }
    
    return repartition

def analyser_couverture_geographique(df_filtre):
    """Analyse la couverture géographique par région"""
    if df_filtre.empty:
        return {}
    
    # Analyse par région
    regions_stats = df_filtre.groupby('region').agg({
        'site_id': 'count',
        'avis': lambda x: (x == 'favorable').sum()
    }).reset_index()
    regions_stats.columns = ['region', 'total_anf', 'anf_favorables']
    regions_stats['taux_approbation'] = round((regions_stats['anf_favorables'] / regions_stats['total_anf']) * 100, 1)
    
    # Analyse par wilaya (top et bottom)
    wilayas_stats = df_filtre.groupby('wilaya').agg({
        'site_id': 'count',
        'avis': lambda x: (x == 'favorable').sum()
    }).reset_index()
    wilayas_stats.columns = ['wilaya', 'total_anf', 'anf_favorables']
    wilayas_stats['taux_approbation'] = round((wilayas_stats['anf_favorables'] / wilayas_stats['total_anf']) * 100, 1)
    
    # Filtrer les wilayas avec au moins 5 ANF pour éviter les biais
    wilayas_stats = wilayas_stats[wilayas_stats['total_anf'] >= 5]
    
    return {
        'regions': regions_stats.to_dict('records'),
        'top_wilayas': wilayas_stats.nlargest(5, 'taux_approbation').to_dict('records'),
        'bottom_wilayas': wilayas_stats.nsmallest(5, 'taux_approbation').to_dict('records')
    }

def calculer_taux_approbation_global(df_filtre):
    """Calcule le taux d'approbation global et les tendances"""
    if df_filtre.empty:
        return {}
    
    # Taux global
    total_anf = len(df_filtre)
    anf_favorables = (df_filtre['avis'] == 'favorable').sum()
    taux_global = round((anf_favorables / total_anf) * 100, 1)
    
    # Évolution par année (si données disponibles)
    evolution_annuelle = df_filtre.groupby('annee').agg({
        'site_id': 'count',
        'avis': lambda x: (x == 'favorable').sum()
    }).reset_index()
    evolution_annuelle.columns = ['annee', 'total_anf', 'anf_favorables']
    evolution_annuelle['taux_approbation'] = round((evolution_annuelle['anf_favorables'] / evolution_annuelle['total_anf']) * 100, 1)
    
    # Calculer la tendance (comparaison dernière vs avant-dernière année)
    tendance = "stable"
    evolution_pct = 0
    
    if len(evolution_annuelle) >= 2:
        derniere_annee = evolution_annuelle.iloc[-1]['taux_approbation']
        avant_derniere_annee = evolution_annuelle.iloc[-2]['taux_approbation']
        evolution_pct = round(derniere_annee - avant_derniere_annee, 1)
        
        if evolution_pct > 2:
            tendance = "hausse"
        elif evolution_pct < -2:
            tendance = "baisse"
    
    return {
        'taux_global': taux_global,
        'tendance': tendance,
        'evolution_pct': evolution_pct,
        'evolution_annuelle': evolution_annuelle.to_dict('records')
    }

def analyser_evolution_temporelle(df_filtre):
    """Analyse l'évolution temporelle et identifie les périodes clés"""
    if df_filtre.empty:
        return {}
    
    # Évolution par trimestre
    evolution_trimestrielle = df_filtre.groupby('periode').size().reset_index()
    evolution_trimestrielle.columns = ['periode', 'nombre_anf']
    evolution_trimestrielle = evolution_trimestrielle.sort_values('periode')
    
    # Identifier les pics et creux
    if len(evolution_trimestrielle) >= 3:
        nombre_anf = evolution_trimestrielle['nombre_anf'].values
        
        # Pic maximum
        max_idx = nombre_anf.argmax()
        pic_max = {
            'periode': evolution_trimestrielle.iloc[max_idx]['periode'],
            'nombre': nombre_anf[max_idx]
        }
        
        # Creux minimum
        min_idx = nombre_anf.argmin()
        creux_min = {
            'periode': evolution_trimestrielle.iloc[min_idx]['periode'],
            'nombre': nombre_anf[min_idx]
        }
        
        # Tendance récente (3 derniers trimestres)
        if len(evolution_trimestrielle) >= 3:
            derniers_3 = nombre_anf[-3:]
            if derniers_3[-1] > derniers_3[0]:
                tendance_recente = "croissance"
            elif derniers_3[-1] < derniers_3[0]:
                tendance_recente = "décroissance"
            else:
                tendance_recente = "stable"
        else:
            tendance_recente = "données insuffisantes"
    else:
        pic_max = creux_min = None
        tendance_recente = "données insuffisantes"
    
    return {
        'evolution_trimestrielle': evolution_trimestrielle.to_dict('records'),
        'pic_max': pic_max,
        'creux_min': creux_min,
        'tendance_recente': tendance_recente
    }

def generer_recommandations_strategiques(analyses):
    """Génère des recommandations basées sur les analyses"""
    recommandations = []
    
    # Recommandations technologiques
    if 'repartition_tech' in analyses:
        tech_data = analyses['repartition_tech']
        
        # Vérifier la part des technologies modernes (4G, 5G)
        tech_modernes = 0
        tech_anciennes = 0
        
        for tech, data in tech_data.items():
            if any(x in tech.lower() for x in ['4g', '5g', 'lte']):
                tech_modernes += data['pourcentage']
            elif any(x in tech.lower() for x in ['2g', '3g', 'gsm', 'umts']):
                tech_anciennes += data['pourcentage']
        
        if tech_anciennes > 50:
            recommandations.append({
                'type': 'technologique',
                'titre': '🔄 Modernisation Technologique Prioritaire',
                'description': f'Avec {tech_anciennes}% de technologies 2G/3G, accélérer la migration vers 4G/5G pour améliorer la qualité de service.',
                'priorite': 'haute'
            })
        elif tech_modernes > 60:
            recommandations.append({
                'type': 'technologique',
                'titre': '✅ Consolidation Technologique',
                'description': f'Bonne adoption des technologies modernes ({tech_modernes}%). Maintenir cette dynamique.',
                'priorite': 'moyenne'
            })
    
    # Recommandations géographiques
    if 'couverture_geo' in analyses:
        geo_data = analyses['couverture_geo']
        
        if 'bottom_wilayas' in geo_data and geo_data['bottom_wilayas']:
            wilayas_faibles = [w['wilaya'] for w in geo_data['bottom_wilayas'][:3]]
            recommandations.append({
                'type': 'geographique',
                'titre': '🗺️ Rééquilibrage Géographique',
                'description': f'Faciliter les ANF dans les wilayas à faible taux d\'approbation : {", ".join(wilayas_faibles)}.',
                'priorite': 'haute'
            })
    
    # Recommandations processus
    if 'taux_approbation' in analyses:
        taux_data = analyses['taux_approbation']
        
        if taux_data['taux_global'] < 70:
            recommandations.append({
                'type': 'processus',
                'titre': '⚡ Optimisation des Processus',
                'description': f'Taux d\'approbation de {taux_data["taux_global"]}%. Digitaliser et simplifier les procédures.',
                'priorite': 'haute'
            })
        elif taux_data['tendance'] == 'baisse':
            recommandations.append({
                'type': 'processus',
                'titre': '📉 Surveillance des Tendances',
                'description': f'Baisse du taux d\'approbation ({taux_data["evolution_pct"]}%). Analyser les causes.',
                'priorite': 'moyenne'
            })
    
    # Recommandation collaboration (toujours présente)
    recommandations.append({
        'type': 'collaboration',
        'titre': '🤝 Collaboration Interinstitutionnelle',
        'description': 'Créer un groupe de travail ANF/opérateurs pour fluidifier les échanges et anticiper les besoins.',
        'priorite': 'moyenne'
    })
    
    return recommandations

def generer_objectifs_2025():
    """Génère les objectifs chiffrés pour 2025"""
    return [
        {
            'titre': '+15% Sites 4G/5G',
            'description': 'Augmenter la part des technologies modernes',
            'valeur_actuelle': '45%',
            'valeur_cible': '60%',
            'icone': '📶',
            'couleur': '#28a745'
        },
        {
            'titre': '-40% Délai Traitement',
            'description': 'Réduire les délais moyens de traitement des ANF',
            'valeur_actuelle': '120 jours',
            'valeur_cible': '72 jours',
            'icone': '⚡',
            'couleur': '#ffc107'
        },
        {
            'titre': '80% Taux Approbation',
            'description': 'Atteindre un taux d\'approbation de référence',
            'valeur_actuelle': '68%',
            'valeur_cible': '80%',
            'icone': '✅',
            'couleur': '#17a2b8'
        },
        {
            'titre': '95% Couverture Numérique',
            'description': 'Étendre la couverture dans toutes les régions',
            'valeur_actuelle': '78%',
            'valeur_cible': '95%',
            'icone': '🌐',
            'couleur': '#6f42c1'
        }
    ]

def creer_timeline_historique():
    """Crée la timeline des évolutions historiques"""
    return [
        {
            'periode': '2009-2011',
            'titre': 'Vague 2G',
            'description': 'Déploiement massif de la 2G dans les zones urbaines. Foundation du réseau mobile algérien.',
            'couleur': '#dc3545',
            'icone': '📱'
        },
        {
            'periode': '2012-2014',
            'titre': 'Consolidation',
            'description': 'Stabilisation du marché 2G et préparation de l\'arrivée de la 3G.',
            'couleur': '#fd7e14',
            'icone': '🔧'
        },
        {
            'periode': '2015-2017',
            'titre': '3G & Zones Rurales',
            'description': 'Lancement de la 3G et extension de la couverture vers les zones rurales.',
            'couleur': '#ffc107',
            'icone': '🌾'
        },
        {
            'periode': '2018-2019',
            'titre': 'Densification',
            'description': 'Densification du réseau et amélioration de la qualité de service.',
            'couleur': '#20c997',
            'icone': '📈'
        },
        {
            'periode': '2020-2021',
            'titre': 'Ralentissement COVID',
            'description': 'Impact de la pandémie sur les déploiements et les processus d\'approbation.',
            'couleur': '#6c757d',
            'icone': '😷'
        },
        {
            'periode': '2022-2024',
            'titre': 'Reprise & 4G',
            'description': 'Reprise post-COVID, accélération de la 4G et préparation de la 5G.',
            'couleur': '#0d6efd',
            'icone': '🚀'
        }
    ]

def generer_rapport_pdf(df_filtre, analyses, recommandations):
    """Génère un rapport PDF complet avec les analyses"""
    # Créer un fichier temporaire
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    pdf_path = temp_file.name
    
    # Créer le document PDF
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []
    
    # Titre du rapport
    title = Paragraph("Rapport d'Analyse ANF Complet", styles['Title'])
    story.append(title)
    story.append(Spacer(1, 12))
    
    # Date de génération
    date_str = f"Généré le {datetime.now().strftime('%d/%m/%Y à %H:%M')}"
    date_para = Paragraph(date_str, styles['Normal'])
    story.append(date_para)
    story.append(Spacer(1, 20))
    
    # Résumé des données
    story.append(Paragraph("Résumé des Données", styles['Heading2']))
    summary_data = [
        ["Total ANF", str(len(df_filtre))],
        ["Technologies uniques", str(df_filtre['technologie'].nunique())],
        ["Régions couvertes", str(df_filtre['region'].nunique())],
        ["Wilayas couvertes", str(df_filtre['wilaya'].nunique())],
        ["Taux d'approbation", f"{analyses['taux_approbation']['taux_global']}%"]
    ]
    summary_table = Table(summary_data, colWidths=[200, 100])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 14),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTSIZE', (0, 1), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 20))
    
    # Répartition technologique
    story.append(Paragraph("Répartition Technologique", styles['Heading2']))
    tech_data = [["Technologie", "Nombre", "Pourcentage"]]
    for tech, data in analyses['repartition_tech'].items():
        tech_data.append([tech, str(data['count']), f"{data['pourcentage']}%"])
    
    tech_table = Table(tech_data, colWidths=[200, 100, 100])
    tech_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 12),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(tech_table)
    story.append(Spacer(1, 20))
    
    # Recommandations
    story.append(Paragraph("Recommandations Stratégiques", styles['Heading2']))
    for i, rec in enumerate(recommandations):
        story.append(Paragraph(f"{i+1}. {rec['titre']}", styles['Heading3']))
        story.append(Paragraph(rec['description'], styles['Normal']))
        story.append(Spacer(1, 10))
    
    # Générer le PDF
    doc.build(story)
    
    return pdf_path

# =============================================================================
# CSS ET INTERFACE
# =============================================================================

def load_css():
    """Charge les styles CSS"""
    st.markdown("""
    <style>
    .main {
        padding: 1rem;
    }
    .upload-zone {
        border: 2px dashed #0d6efd;
        border-radius: 10px;
        padding: 3rem;
        text-align: center;
        background-color: rgba(13, 110, 253, 0.05);
        margin-bottom: 2rem;
    }
    .kpi-box {
        padding: 1rem;
        border-radius: 8px;
        background-color: rgba(13, 110, 253, 0.1);
        text-align: center;
        margin-bottom: 1rem;
    }
    .kpi-value {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0;
        color: #0d6efd;
    }
    .kpi-label {
        font-size: 0.9rem;
        opacity: 0.7;
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)

load_css()

# =============================================================================
# INITIALISATION DU STATE
# =============================================================================

if "page" not in st.session_state:
    st.session_state.page = "upload"
if "data" not in st.session_state:
    st.session_state.data = None
if "processed_data" not in st.session_state:
    st.session_state.processed_data = None
if "column_mapping" not in st.session_state:
    st.session_state.column_mapping = {}
if "filters" not in st.session_state:
    st.session_state.filters = {}

# =============================================================================
# FONCTIONS DE L'APPLICATION
# =============================================================================

def valider_colonnes(df):
    """Valide et map les colonnes"""
    df_normalise, colonnes_mappees, colonnes_manquantes = mapper_colonnes(df)
    validation_ok = len(colonnes_manquantes) == 0
    return validation_ok, colonnes_mappees, colonnes_manquantes, df_normalise

def afficher_mappings_colonnes(colonnes_mappees):
    """Affiche les mappings de colonnes"""
    if colonnes_mappees:
        st.info("✅ Colonnes automatiquement mappées :")
        mapping_data = [[col_std, col_orig] for col_std, col_orig in colonnes_mappees.items()]
        mapping_df = pd.DataFrame(mapping_data, columns=["Nom standard", "Nom dans le fichier"])
        st.dataframe(mapping_df, use_container_width=True, hide_index=True)

def afficher_preview(df):
    """Affiche la prévisualisation des données"""
    st.subheader("Prévisualisation des données")
    st.dataframe(df.head(10), use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Nombre de lignes", df.shape[0])
    col2.metric("Nombre de colonnes", df.shape[1])
    
    if 'date_depot_anf' in df.columns and not df['date_depot_anf'].isna().all():
        try:
            date_min = df['date_depot_anf'].min().strftime('%d/%m/%Y')
            date_max = df['date_depot_anf'].max().strftime('%d/%m/%Y')
            col3.metric("Période", f"{date_min} - {date_max}")
        except:
            col3.metric("Période", "Non disponible")
    else:
        col3.metric("Période", "Non disponible")

# =============================================================================
# PAGES
# =============================================================================

def page_upload():
    """Page d'upload du fichier"""
    st.title("📊 Application d'Analyse de Données ANF")
    
    st.markdown("""
    <div class="upload-zone">
        <h3>📁 Glissez et déposez votre fichier Excel</h3>
        <p>ou utilisez le bouton ci-dessous</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choisissez un fichier Excel (.xlsx)", type="xlsx")
    
    if uploaded_file is not None:
        try:
            # Lecture du fichier
            df = pd.read_excel(uploaded_file)
            
            st.success(f"✅ Fichier lu avec succès! {df.shape[0]} lignes, {df.shape[1]} colonnes")
            
            # Validation et mapping
            validation_ok, colonnes_mappees, colonnes_manquantes, df_normalise = valider_colonnes(df)
            
            st.session_state.column_mapping = colonnes_mappees
            
            if validation_ok:
                # Mapping réussi
                afficher_mappings_colonnes(colonnes_mappees)
                
                # Conversion des dates
                df_normalise['date_depot_anf'] = pd.to_datetime(df_normalise['date_depot_anf'], errors='coerce')
                
                # Stockage
                st.session_state.data = df_normalise
                
                st.success("🎉 Toutes les colonnes requises ont été identifiées!")
                
                # Prévisualisation
                afficher_preview(df_normalise)
                
                # Bouton pour continuer
                if st.button("🚀 Procéder à l'analyse des données", type="primary"):
                    st.session_state.page = "analyse"
                    st.rerun()
                    
            else:
                # Mapping incomplet
                st.error(f"❌ Colonnes manquantes : {', '.join(colonnes_manquantes)}")
                
                # Suggestions
                suggestions = suggerer_corrections_colonnes(colonnes_manquantes, df)
                if suggestions:
                    st.warning("💡 Suggestions :")
                    for col_manquante, cols_suggerees in suggestions.items():
                        st.write(f"• Pour **{col_manquante}** : {', '.join(cols_suggerees)}")
                
                # Mapping manuel
                st.subheader("🔧 Mapping manuel des colonnes")
                mapping_manuel = {}
                colonnes_disponibles = list(df.columns)
                
                for col_manquante in colonnes_manquantes:
                    selected_col = st.selectbox(
                        f"Colonne pour '{col_manquante}' :",
                        options=["Sélectionner..."] + colonnes_disponibles,
                        key=f"map_{col_manquante}"
                    )
                    if selected_col != "Sélectionner...":
                        mapping_manuel[col_manquante] = selected_col
                
                # Appliquer le mapping manuel
                if len(mapping_manuel) == len(colonnes_manquantes):
                    if st.button("✅ Appliquer le mapping manuel", type="primary"):
                        # Créer le DataFrame mappé
                        df_mappe = df.copy()
                        
                        # Appliquer tous les mappings
                        all_mappings = {**colonnes_mappees, **mapping_manuel}
                        for col_std, col_orig in all_mappings.items():
                            if col_orig in df_mappe.columns:
                                df_mappe.rename(columns={col_orig: col_std}, inplace=True)
                        
                        # Vérifier que toutes les colonnes sont présentes
                        required_cols = ["site_id", "technologie", "region", "wilaya", "date_depot_anf", "avis"]
                        if all(col in df_mappe.columns for col in required_cols):
                            st.session_state.column_mapping = all_mappings
                            df_mappe['date_depot_anf'] = pd.to_datetime(df_mappe['date_depot_anf'], errors='coerce')
                            st.session_state.data = df_mappe
                            
                            st.success("🎉 Mapping manuel appliqué avec succès!")
                            
                            # Afficher les mappings
                            st.subheader("Mappings utilisés")
                            mapping_data = [[k, v] for k, v in all_mappings.items()]
                            mapping_df = pd.DataFrame(mapping_data, columns=["Nom standard", "Nom dans le fichier"])
                            st.dataframe(mapping_df, use_container_width=True, hide_index=True)
                            
                            # Prévisualisation
                            afficher_preview(df_mappe)
                            
                            # Bouton pour continuer
                            if st.button("🚀 Procéder à l'analyse", key="proceed_manual", type="primary"):
                                st.session_state.page = "analyse"
                                st.rerun()
                        else:
                            st.error("❌ Erreur dans le mapping. Veuillez vérifier vos sélections.")
        
        except Exception as e:
            st.error(f"❌ Erreur lors de la lecture du fichier : {str(e)}")
            st.info("Assurez-vous que votre fichier est un Excel (.xlsx) valide.")
    else:
        st.info("👆 Veuillez charger un fichier Excel contenant vos données ANF.")
        
        # Information sur le format attendu
        with st.expander("ℹ️ Format de fichier attendu"):
            st.markdown("""
            Votre fichier Excel doit contenir des colonnes équivalentes à :
            - **site_id** : Identifiant du site
            - **technologie** : Type de technologie
            - **region** : Région géographique
            - **wilaya** : Wilaya concernée
            - **date_depot_anf** : Date de dépôt
            - **avis** : Avis donné (favorable/défavorable/ok/en instance)
            
            L'application peut reconnaître automatiquement des variantes de ces noms.
            """)

def page_analyse():
    """Page d'analyse des données avec filtres dynamiques"""
    st.title("📊 Tableau de Bord ANF")
    
    # Traiter les données si nécessaire
    if st.session_state.processed_data is None and st.session_state.data is not None:
        with st.spinner("Traitement des données en cours..."):
            st.session_state.processed_data = pretraiter_donnees(st.session_state.data)
    
    if st.session_state.processed_data is None:
        st.error("❌ Aucune donnée disponible.")
        if st.button("↩️ Retour à l'upload"):
            st.session_state.page = "upload"
            st.rerun()
        return
    
    processed_data = st.session_state.processed_data
    df = processed_data['df']
    valeurs_uniques = processed_data['valeurs_uniques']
    
    # =============================================================================
    # SECTION FILTRES DYNAMIQUES
    # =============================================================================
    
    st.markdown('<div class="filter-section">', unsafe_allow_html=True)
    st.subheader("🔍 Filtres")
    
    # Organisation des filtres en colonnes
    col_filter1, col_filter2, col_filter3, col_filter4, col_filter5 = st.columns(5)
    
    with col_filter1:
        technologie_filter = st.multiselect(
            "Technologie",
            options=valeurs_uniques['technologie'],
            default=[],
            key="tech_filter"
        )
    
    with col_filter2:
        region_filter = st.multiselect(
            "Région",
            options=valeurs_uniques['region'],
            default=[],
            key="region_filter"
        )
    
    with col_filter3:
        wilaya_filter = st.multiselect(
            "Wilaya",
            options=valeurs_uniques['wilaya'],
            default=[],
            key="wilaya_filter"
        )
    
    with col_filter4:
        # Filtre avis avec ordre spécifique
        avis_filter = st.multiselect(
            "Avis",
            options=['favorable', 'défavorable', 'en instance', 'Non spécifié'],
            default=[],
            key="avis_filter"
        )
    
    with col_filter5:
        periode_filter = st.multiselect(
            "Période",
            options=valeurs_uniques['periode'],
            default=[],
            key="periode_filter"
        )
    
    # Bouton pour réinitialiser les filtres
    if st.button("🔄 Réinitialiser tous les filtres"):
        st.session_state.filters = {}
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Mise à jour des filtres dans le session state
    st.session_state.filters = {
        'technologie': technologie_filter,
        'region': region_filter,
        'wilaya': wilaya_filter,
        'avis': avis_filter,
        'periode': periode_filter
    }
    
    # Application des filtres aux données
    df_filtre = appliquer_filtres(df, st.session_state.filters)
    
    # Affichage du nombre de résultats
    if len(df_filtre) != len(df):
        st.info(f"📊 Affichage de **{len(df_filtre)}** résultats sur **{len(df)}** total avec les filtres appliqués")
    
    # =============================================================================
    # KPIS DYNAMIQUES
    # =============================================================================
    
    st.subheader("📈 Indicateurs Clés")
    kpis = create_kpi_metrics(df_filtre)
    
    kpi_cols = st.columns(len(kpis))
    for i, (label, value) in enumerate(kpis.items()):
        with kpi_cols[i]:
            st.markdown(f"""
            <div class="kpi-box">
                <div class="kpi-value">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>
            """, unsafe_allow_html=True)
    
    # =============================================================================
    # VISUALISATIONS PRINCIPALES FILTRABLES
    # =============================================================================
    
    st.subheader("📊 Visualisations Principales")
    
    # Première ligne : Technologie + Top Wilayas Favorables
    col1, col2 = st.columns(2)
    
    with col1:
        fig_tech = create_technology_pie_chart(df_filtre)
        st.plotly_chart(fig_tech, use_container_width=True)
        
        # Ajouter le tableau récapitulatif des technologies
        tech_table = create_technology_summary_table(df_filtre)
        if tech_table is not None:
            st.subheader("📋 Tableau récapitulatif des technologies")
            st.dataframe(tech_table, use_container_width=True)
    
    with col2:
        fig_wilayas = create_top_wilayas_favorable_chart(df_filtre)
        st.plotly_chart(fig_wilayas, use_container_width=True)
    
    # =============================================================================
    # NOUVELLES VISUALISATIONS DEMANDÉES
    # =============================================================================
    
    st.subheader("🗺️ Répartition Géographique des Technologies")
    
    # Deuxième ligne : Technologies par Région + Technologies par Wilaya
    col3, col4 = st.columns(2)
    
    with col3:
        fig_tech_region = create_technology_by_region_chart(df_filtre)
        st.plotly_chart(fig_tech_region, use_container_width=True)
    
    with col4:
        fig_tech_wilaya = create_technology_by_wilaya_chart(df_filtre)
        st.plotly_chart(fig_tech_wilaya, use_container_width=True)
    
    # =============================================================================
    # ÉVOLUTION TEMPORELLE EN COURBES
    # =============================================================================
    
    st.subheader("📅 Évolution Temporelle")
    fig_evolution = create_evolution_line_chart(df_filtre)
    st.plotly_chart(fig_evolution, use_container_width=True)
    
    # =============================================================================
    # DONNÉES DÉTAILLÉES AVEC EXPORT
    # =============================================================================
    
    st.subheader("📋 Données Détaillées")
    
    # Afficher les données filtrées
    if not df_filtre.empty:
        st.dataframe(df_filtre, use_container_width=True)
        
        # Boutons d'export
        col_export1, col_export2, col_export3 = st.columns(3)
        
        with col_export1:
            # Export CSV
            csv = df_filtre.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="💾 Télécharger CSV",
                data=csv,
                file_name=f"donnees_anf_filtrees_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
        
        with col_export2:
            # Export Excel
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                df_filtre.to_excel(writer, index=False, sheet_name='Données ANF')
            output.seek(0)
            
            st.download_button(
                label="📊 Télécharger Excel",
                data=output.getvalue(),
                file_name=f"donnees_anf_filtrees_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        
        with col_export3:
            if st.button("🔄 Actualiser l'analyse"):
                st.session_state.processed_data = pretraiter_donnees(st.session_state.data)
                st.rerun()
    
    else:
        st.warning("⚠️ Aucune donnée ne correspond aux filtres sélectionnés. Veuillez ajuster vos critères.")
    
    # =============================================================================
    # STATISTIQUES AVANCÉES (OPTIONNEL)
    # =============================================================================
    
    with st.expander("📈 Statistiques Avancées"):
        if not df_filtre.empty:
            col_stat1, col_stat2, col_stat3 = st.columns(3)
            
            with col_stat1:
                st.markdown("**Distribution des Avis**")
                avis_stats = df_filtre['avis'].value_counts()
                for avis, count in avis_stats.items():
                    pct = round(count / len(df_filtre) * 100, 1)
                    st.text(f"{avis}: {count} ({pct}%)")
            
            with col_stat2:
                st.markdown("**Top 3 Technologies**")
                tech_stats = df_filtre['technologie'].value_counts().head(3)
                for tech, count in tech_stats.items():
                    pct = round(count / len(df_filtre) * 100, 1)
                    st.text(f"{tech}: {count} ({pct}%)")
            
            with col_stat3:
                st.markdown("**Top 3 Régions**")
                region_stats = df_filtre['region'].value_counts().head(3)
                for region, count in region_stats.items():
                    pct = round(count / len(df_filtre) * 100, 1)
                    st.text(f"{region}: {count} ({pct}%)")
    
    # Bouton pour accéder aux conclusions
    st.markdown("---")
    col_nav1, col_nav2, col_nav3 = st.columns(3)
    
    with col_nav2:
        if st.button("📋 Voir Conclusions & Recommandations", type="primary", use_container_width=True):
            st.session_state.page = "conclusions"
            st.rerun()

def page_conclusions_recommandations():
    """Page des conclusions et recommandations"""
    st.title("📋 Conclusions & Recommandations")
    
    # Vérifier que les données sont disponibles
    if st.session_state.processed_data is None:
        st.error("❌ Aucune donnée disponible. Veuillez d'abord charger et analyser un fichier.")
        if st.button("↩️ Retour à l'upload"):
            st.session_state.page = "upload"
            st.rerun()
        return
    
    # Récupérer les données (filtrées ou complètes selon le contexte)
    df = st.session_state.processed_data['df']
    
    # Appliquer les filtres si ils existent
    if st.session_state.filters:
        df_analyse = appliquer_filtres(df, st.session_state.filters)
        if len(df_analyse) != len(df):
            st.info(f"📊 Analyse basée sur **{len(df_analyse)}** ANF filtrées sur **{len(df)}** total")
    else:
        df_analyse = df
    
    # =============================================================================
    # 1. SYNTHÈSE AUTOMATIQUE
    # =============================================================================
    
    st.header("📊 Synthèse Analytique")
    
    # Analyses des données
    analyses = {
        'repartition_tech': analyser_repartition_technologique(df_analyse),
        'couverture_geo': analyser_couverture_geographique(df_analyse),
        'taux_approbation': calculer_taux_approbation_global(df_analyse),
        'evolution_temp': analyser_evolution_temporelle(df_analyse)
    }
    
    # Affichage de la synthèse en colonnes
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔧 Répartition Technologique")
        
        if analyses['repartition_tech']:
            for tech, data in analyses['repartition_tech'].items():
                # Couleur selon le type de technologie
                if any(x in tech.lower() for x in ['5g']):
                    color = "🟢"
                elif any(x in tech.lower() for x in ['4g', 'lte']):
                    color = "🔵"
                elif any(x in tech.lower() for x in ['3g']):
                    color = "🟡"
                else:
                    color = "🔴"
                
                st.markdown(f"{color} **{tech}** : {data['pourcentage']}% ({data['count']} sites)")
        else:
            st.info("Aucune donnée technologique disponible")
        
        st.subheader("📈 Taux d'Approbation")
        if analyses['taux_approbation']:
            taux_data = analyses['taux_approbation']
            
            st.metric(
                "Taux Global",
                f"{taux_data['taux_global']}%",
                delta=f"{taux_data['evolution_pct']}%" if taux_data['evolution_pct'] != 0 else None
            )
            
            # Tendance
            if taux_data['tendance'] == 'hausse':
                st.success(f"📈 Tendance à la hausse (+{taux_data['evolution_pct']}%)")
            elif taux_data['tendance'] == 'baisse':
                st.error(f"📉 Tendance à la baisse ({taux_data['evolution_pct']}%)")
            else:
                st.info("➡️ Tendance stable")
    
    with col2:
        st.subheader("🗺️ Couverture Géographique")
        
        if analyses['couverture_geo']['regions']:
            # Top 3 régions par taux d'approbation
            regions_sorted = sorted(analyses['couverture_geo']['regions'], 
                                  key=lambda x: x['taux_approbation'], reverse=True)
            
            st.markdown("**🏆 Top 3 Régions :**")
            for i, region in enumerate(regions_sorted[:3]):
                medal = ["🥇", "🥈", "🥉"][i]
                st.markdown(f"{medal} {region['region']} : {region['taux_approbation']}%")
            
            st.markdown("**⚠️ Régions à améliorer :**")
            for region in regions_sorted[-2:]:
                st.markdown(f"🔴 {region['region']} : {region['taux_approbation']}%")
        
        st.subheader("⏱️ Évolution Temporelle")
        if analyses['evolution_temp']['tendance_recente']:
            if analyses['evolution_temp']['tendance_recente'] == 'croissance':
                st.success("📈 Tendance récente : Croissance")
            elif analyses['evolution_temp']['tendance_recente'] == 'décroissance':
                st.warning("📉 Tendance récente : Décroissance")
            else:
                st.info("➡️ Tendance récente : Stable")
            
            if analyses['evolution_temp']['pic_max']:
                st.info(f"🔝 Pic maximum : {analyses['evolution_temp']['pic_max']['nombre']} ANF en {analyses['evolution_temp']['pic_max']['periode']}")
    
    # =============================================================================
    # 2. RECOMMANDATIONS STRATÉGIQUES
    # =============================================================================
    
    st.header("💡 Recommandations Stratégiques")
    
    recommandations = generer_recommandations_strategiques(analyses)
    
    # Afficher les recommandations par priorité
    recommandations_haute = [r for r in recommandations if r['priorite'] == 'haute']
    recommandations_moyenne = [r for r in recommandations if r['priorite'] == 'moyenne']
    
    if recommandations_haute:
        st.subheader("🔥 Priorité Haute")
        for rec in recommandations_haute:
            st.error(f"**{rec['titre']}**\n\n{rec['description']}")
    
    if recommandations_moyenne:
        st.subheader("⚡ Priorité Moyenne")
        for rec in recommandations_moyenne:
            st.warning(f"**{rec['titre']}**\n\n{rec['description']}")
    
    # =============================================================================
    # 3. OBJECTIFS CHIFFRÉS 2025
    # =============================================================================
    
    st.header("🎯 Objectifs Chiffrés pour 2025")
    
    objectifs = generer_objectifs_2025()
    
    # Affichage en cartes KPI
    cols = st.columns(2)
    
    for i, objectif in enumerate(objectifs):
        with cols[i % 2]:
            st.markdown(f"""
            <div style="
                border: 2px solid {objectif['couleur']};
                border-radius: 10px;
                padding: 20px;
                margin: 10px 0;
                background: linear-gradient(135deg, {objectif['couleur']}15, {objectif['couleur']}05);
            ">
                <h3 style="color: {objectif['couleur']}; margin: 0;">
                    {objectif['icone']} {objectif['titre']}
                </h3>
                <p style="margin: 10px 0; color: #666;">
                    {objectif['description']}
                </p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold;">Actuel: {objectif['valeur_actuelle']}</span>
                    <span style="color: {objectif['couleur']}; font-weight: bold; font-size: 1.2em;">
                        → {objectif['valeur_cible']}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    # =============================================================================
    # 4. TIMELINE HISTORIQUE
    # =============================================================================
    
    st.header("📅 Timeline des Évolutions Marquantes")
    
    timeline_data = creer_timeline_historique()
    
    # Affichage de la timeline horizontale
    for i, periode in enumerate(timeline_data):
        col_icon, col_content = st.columns([1, 5])
        
        with col_icon:
            st.markdown(f"""
            <div style="
                background-color: {periode['couleur']};
                color: white;
                border-radius: 50%;
                width: 60px;
                height: 60px;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 1.5em;
                margin: 10px auto;
            ">
                {periode['icone']}
            </div>
            """, unsafe_allow_html=True)
        
        with col_content:
            st.markdown(f"""
            <div style="
                border-left: 3px solid {periode['couleur']};
                padding-left: 20px;
                margin: 10px 0;
                min-height: 80px;
            ">
                <h4 style="color: {periode['couleur']}; margin: 0;">
                    {periode['periode']} - {periode['titre']}
                </h4>
                <p style="margin: 5px 0; color: #555;">
                    {periode['description']}
                </p>
            </div>
            """, unsafe_allow_html=True)
        
        # Ajouter une ligne de connexion sauf pour le dernier élément
        if i < len(timeline_data) - 1:
            st.markdown("""
            <div style="
                height: 20px;
                border-left: 2px dashed #ccc;
                margin-left: 30px;
            "></div>
            """, unsafe_allow_html=True)
    
    # =============================================================================
    # 5. ACTIONS ET EXPORT
    # =============================================================================
    
    st.header("📤 Actions")
    
    col_action1, col_action2, col_action3 = st.columns(3)
    
    with col_action1:
        if st.button("📊 Retour aux Analyses", type="secondary"):
            st.session_state.page = "analyse"
            st.rerun()
    
    with col_action2:
        if st.button("📄 Générer Rapport Complet"):
            with st.spinner("Génération du rapport en cours..."):
                # Générer le rapport PDF
                pdf_path = generer_rapport_pdf(df_analyse, analyses, recommandations)
                
                # Lire le fichier PDF généré
                with open(pdf_path, "rb") as f:
                    pdf_data = f.read()
                
                # Proposer le téléchargement
                st.success("Rapport généré avec succès!")
                st.download_button(
                    label="📥 Télécharger le Rapport PDF",
                    data=pdf_data,
                    file_name=f"rapport_anf_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                    mime="application/pdf"
                )
                
                # Nettoyer le fichier temporaire
                os.unlink(pdf_path)
    
    with col_action3:
        if st.button("📥 Nouveau Fichier"):
            st.session_state.page = "upload"
            st.session_state.data = None
            st.session_state.processed_data = None
            st.session_state.filters = {}
            st.rerun()

# =============================================================================
# APPLICATION PRINCIPALE
# =============================================================================

def main():
    """Fonction principale avec navigation mise à jour"""
    
    # Navigation dans la sidebar
    with st.sidebar:
        st.title("🏠 Navigation")
        
        # Menu de navigation selon l'état des données
        if st.session_state.data is not None:
            # Si des données sont chargées, montrer toutes les pages
            pages_disponibles = {
                "📊 Analyses": "analyse",
                "📋 Conclusions": "conclusions"
            }
            
            # Page par défaut si données disponibles
            if st.session_state.page == "upload":
                st.session_state.page = "analyse"
        else:
            # Si pas de données, seulement la page upload
            pages_disponibles = {
                "📁 Upload": "upload"
            }
        
        # Affichage du menu de navigation
        st.markdown("### 📑 Pages disponibles")
        for nom_page, code_page in pages_disponibles.items():
            if st.button(nom_page, use_container_width=True):
                st.session_state.page = code_page
                st.rerun()
        
        # Indicateur de page actuelle
        if st.session_state.page == "upload":
            st.info("📍 Page actuelle : Upload")
        elif st.session_state.page == "analyse":
            st.info("📍 Page actuelle : Analyses")
        elif st.session_state.page == "conclusions":
            st.info("📍 Page actuelle : Conclusions")
        
        # Informations sur l'état des données
        st.markdown("---")
        st.markdown("### 📊 État des Données")
        
        if st.session_state.data is not None:
            st.success("✅ Données chargées")
            st.info(f"📈 {len(st.session_state.data)} lignes")
            
            if st.session_state.processed_data is not None:
                st.success("✅ Données traitées")
            
            # Afficher les filtres actifs
            if st.session_state.filters:
                filtres_actifs = [k for k, v in st.session_state.filters.items() if v]
                if filtres_actifs:
                    st.warning(f"🔍 Filtres actifs : {len(filtres_actifs)}")
        else:
            st.warning("⏳ Aucune donnée")
        
        # Actions rapides
        st.markdown("---")
        st.markdown("### ⚡ Actions Rapides")
        
        if st.button("🗑️ Réinitialiser tout", use_container_width=True):
            # Réinitialiser complètement l'application
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # Contenu principal selon la page
    if st.session_state.page == "upload":
        page_upload()
    elif st.session_state.page == "analyse":
        page_analyse()
    elif st.session_state.page == "conclusions":
        page_conclusions_recommandations()
    else:
        # Page par défaut
        st.session_state.page = "upload"
        page_upload()

# =============================================================================
# POINT D'ENTRÉE DE L'APPLICATION
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"🚨 Erreur inattendue : {str(e)}")
        st.markdown("---")
        
        with st.expander("🔍 Détails de l'erreur (pour le débogage)"):
            st.code(f"""
            Type d'erreur: {type(e).__name__}
            Message: {str(e)}
            
            État de l'application:
            - Page actuelle: {st.session_state.get('page', 'Non définie')}
            - Données chargées: {'Oui' if st.session_state.get('data') is not None else 'Non'}
            - Données traitées: {'Oui' if st.session_state.get('processed_data') is not None else 'Non'}
            """)
        
        st.markdown("### 🔧 Actions possibles:")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🔄 Réessayer"):
                st.rerun()
        
        with col2:
            if st.button("🏠 Retour à l'accueil"):
                st.session_state.page = "upload"
                st.rerun()
        
        with col3:
            if st.button("🗑️ Réinitialiser"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()

# =============================================================================
# CONFIGURATION FINALE
# =============================================================================

# Footer informatif
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.8em;'>
    📊 Application d'Analyse ANF | Version 3.0 | 
    Développée avec Streamlit 🚀 | Module Conclusions & Recommandations ✨
</div>
""", unsafe_allow_html=True)
