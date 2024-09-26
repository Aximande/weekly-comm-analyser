import streamlit as st
import pandas as pd
from openai import OpenAI
import re
from datetime import datetime
import os
import tiktoken

# Configuration de la clé API OpenAI
openai_api_key = st.secrets["openai_api_key"]

if not openai_api_key:
    st.error("La clé API OpenAI n'est pas configurée. Veuillez définir votre clé API.")
    st.stop()
else:
    client = OpenAI(api_key=openai_api_key)

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Analyse des Commentaires - Brut",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Afficher le logo de "Brut"
st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/5/50/Brut_logo.svg/1200px-Brut_logo.svg.png", width=150)

# Titre de l'application
st.title("Brut.Edito: Analyse des Questions Non Répondues de nos publications Brut")
st.write("Bienvenue dans l'outil d'analyse des commentaires pour les journalistes de **Brut**.")

# Explications pour les journalistes
st.markdown("""
Cet outil vous permet d'identifier rapidement les questions non répondues dans les commentaires de vos vidéos publiées sur les réseaux sociaux. En se concentrant sur les vidéos les plus commentées ou les plus récentes, vous pouvez prioriser les sujets qui suscitent le plus d'engagement de la part de votre audience.

**Instructions :**

1. **Charger le fichier CSV** contenant les commentaires de vos vidéos.
2. **Sélectionner les paramètres d'analyse** selon vos besoins.
3. **Lancer l'analyse** pour obtenir les questions non répondues les plus pertinentes.
4. **Exporter les résultats** pour les utiliser dans votre travail journalistique.

*Veuillez noter que ce processus peut prendre quelques minutes en fonction du nombre de vidéos sélectionnées.*
""")

# Avertissements et conseils
st.info("**Conseil :** Pour une analyse plus rapide, limitez le nombre de vidéos sélectionnées ou utilisez des filtres pour cibler les vidéos les plus importantes.")
st.warning("**Avertissement :** Assurez-vous que votre fichier CSV est correctement formaté et que toutes les colonnes nécessaires sont présentes.")

# Chargement du fichier CSV
uploaded_file = st.file_uploader("Veuillez charger le fichier CSV des commentaires :", type="csv")

if uploaded_file is not None:
    # Lecture du fichier CSV avec encodage UTF-8
    try:
        df = pd.read_csv(uploaded_file, encoding='utf-8')
        st.success("Fichier CSV chargé avec succès.")
    except Exception as e:
        st.error(f"Erreur lors du chargement du fichier CSV : {e}")
        st.stop()

    # Vérification des colonnes nécessaires
    required_columns = ['platform_code', 'publication_id', 'comment_id', 'published_at', 'comment_text', 'user_id', 'parent_id', 'is_an_answer', 'answer_count', 'like_count']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"Les colonnes suivantes sont manquantes dans le fichier CSV : {', '.join(missing_columns)}")
        st.stop()

    # Nettoyage des données
    df = df[df['comment_text'].notnull()].reset_index(drop=True)
    df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')

    # Calculer le nombre de commentaires par vidéo
    commentaires_par_video = df.groupby('publication_id').size().reset_index(name='nombre_de_commentaires')

    # Sélection des vidéos à analyser
    st.subheader("Paramètres de sélection des vidéos")
    st.write("Utilisez les options ci-dessous pour sélectionner les vidéos à analyser.")
    tri_option = st.selectbox("Trier les vidéos par :", options=['Nombre de commentaires', 'Date de publication'])
    N = st.slider("Nombre de vidéos à analyser :", min_value=1, max_value=150, value=50, help="Nombre de vidéos les plus importantes à analyser.")

    if tri_option == 'Nombre de commentaires':
        # Trier les vidéos par nombre de commentaires décroissant
        videos_triees = commentaires_par_video.sort_values(by='nombre_de_commentaires', ascending=False)
    else:
        # Obtenir la date de publication pour chaque vidéo
        dates_publication = df[['publication_id', 'published_at']].drop_duplicates()
        videos_triees = dates_publication.sort_values(by='published_at', ascending=False)
        videos_triees = videos_triees.merge(commentaires_par_video, on='publication_id')

    # Sélectionner les N vidéos à analyser
    top_videos = videos_triees.head(N)['publication_id'].tolist()

    st.write(f"Vous avez choisi d'analyser les {N} vidéos triées par **{tri_option.lower()}**.")

    # Afficher les vidéos sélectionnées
    st.markdown("**Vidéos sélectionnées :**")
    st.dataframe(videos_triees.head(N))

    # Filtrer le DataFrame pour les vidéos sélectionnées
    df_selected = df[df['publication_id'].isin(top_videos)]

    # Identification des commentaires potentiellement questions
    def est_potentiellement_question(texte):
        if pd.isnull(texte):
            return False
        if '?' in str(texte):
            return True
        mots_interrogatifs = ['qui', 'quoi', 'quand', 'où', 'pourquoi', 'comment', 'combien', 'lequel', 'laquelle']
        texte_minuscule = str(texte).lower()
        for mot in mots_interrogatifs:
            if re.search(r'\b' + mot + r'\b', texte_minuscule):
                return True
        return False

    # Utiliser .loc pour éviter les avertissements
    df_selected.loc[:, 'potentiellement_question'] = df_selected['comment_text'].apply(est_potentiellement_question)

    # Utiliser 'answer_count' pour identifier les questions non répondues
    if 'answer_count' in df_selected.columns:
        df_selected.loc[:, 'answer_count'] = df_selected['answer_count'].fillna(0)
        df_selected.loc[:, 'non_repondue'] = df_selected.apply(lambda row: row['potentiellement_question'] and row['answer_count'] == 0, axis=1)
    else:
        st.warning("La colonne 'answer_count' n'est pas disponible. Toutes les questions seront considérées comme potentiellement non répondues.")
        df_selected.loc[:, 'non_repondue'] = df_selected['potentiellement_question']

    # Regrouper les commentaires non répondus par vidéo
    commentaires_par_video = {}
    grouped = df_selected.groupby('publication_id')
    for publication_id, group in grouped:
        commentaires_non_repondus = group[group['non_repondue']]['comment_text'].tolist()
        if commentaires_non_repondus:
            commentaires_par_video[publication_id] = commentaires_non_repondus

    # Limiter le nombre total d'appels API
    max_api_calls = 100
    videos_a_analyser = list(commentaires_par_video.keys())[:max_api_calls]

    st.write(f"Nombre total d'appels API prévus : **{len(videos_a_analyser)}**")
    st.write("L'analyse commencera sur les vidéos sélectionnées ci-dessous. Cela peut prendre quelques minutes.")

    # Bouton pour lancer l'analyse
    if st.button("Lancer l'analyse"):
        st.subheader("Analyse des vidéos sélectionnées")
        # Initialiser une liste pour les résultats
        resultats_list = []

        # Initialiser le tokenizer pour le modèle GPT-4o-mini
        encoding = tiktoken.encoding_for_model('gpt-4o-mini')

        for publication_id in videos_a_analyser:
            with st.spinner(f"Analyse de la vidéo {publication_id} en cours..."):
                commentaires = commentaires_par_video[publication_id]

                # Préparer les messages pour l'API Chat
                commentaires_concaténés = '\n'.join(commentaires)

                # Préparer le prompt
                messages = [
                    {"role": "system", "content": "Vous êtes un assistant qui aide à analyser les commentaires sous une vidéo."},
                    {"role": "user", "content": f"""
Les commentaires suivants sont des questions non répondues posées par les spectateurs en français. Veuillez :

1. Identifier les questions les plus récurrentes ou similaires et les résumer.
2. Fournir une liste des principales questions non répondues.

Commentaires non répondus :

{commentaires_concaténés}

Veuillez fournir votre réponse en français.
"""}
                ]

                # Fonction pour compter le nombre de tokens
                def count_tokens(messages):
                    num_tokens = 0
                    for message in messages:
                        content = message['content']
                        num_tokens += len(encoding.encode(content))
                    return num_tokens

                max_tokens_model = 128000  # Pour GPT-4o-mini
                tokens_prompt = count_tokens(messages)

                # Debug logging
                st.write(f"tokens_prompt: {tokens_prompt}, type: {type(tokens_prompt)}")
                st.write(f"max_tokens_model: {max_tokens_model}, type: {type(max_tokens_model)}")

                # Vérifier si le prompt dépasse la longueur maximale
                if tokens_prompt > max_tokens_model - 16384:  # Réserver 16 384 tokens pour la réponse
                    st.warning(f"Le prompt pour la vidéo {publication_id} est trop long ({tokens_prompt} tokens). Les commentaires seront tronqués.")
                    # Calculer le nombre de tokens disponibles pour les commentaires
                    tokens_disponibles = max_tokens_model - 16384 - count_tokens(messages[:1])  # Enlever les tokens du système et de l'instruction

                    # Tronquer les commentaires pour s'adapter
                    commentaires_tokens = encoding.encode(commentaires_concaténés)
                    commentaires_tronqués = encoding.decode(commentaires_tokens[:tokens_disponibles])
                    messages[1]['content'] = f"""
Les commentaires suivants sont des questions non répondues posées par les spectateurs en français. Veuillez :

1. Identifier les questions les plus récurrentes ou similaires et les résumer.
2. Fournir une liste des principales questions non répondues.

Commentaires non répondus :

{commentaires_tronqués}

Veuillez fournir votre réponse en français.
"""
                try:
                    completion = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=messages,
                        max_tokens=16384,  # Sortie maximale de 16 384 tokens
                        temperature=0.7,
                        n=1,
                    )

                    # Récupérer la réponse
                    analyse = completion.choices[0].message.content.strip()

                    # Extraction des sections
                    def extraire_sections(analyse):
                        sections = {}
                        matches = re.findall(r'(\d+)\.\s*(.*?)(?=\n\d+\.|\Z)', analyse, re.DOTALL)
                        for numero, contenu in matches:
                            sections[int(numero)] = contenu.strip()
                        return sections
                    sections = extraire_sections(analyse)

                    # Ajouter les résultats à la liste
                    resultats_list.append({
                        'publication_id': publication_id,
                        'questions_récurrentes': sections.get(1, ''),
                        'principales_questions': sections.get(2, '')
                    })

                    st.success(f"Analyse terminée pour la vidéo {publication_id}")
                except Exception as e:
                    st.error(f"Une erreur est survenue lors de l'analyse de la vidéo {publication_id} : {e}")

        # Créer le DataFrame des résultats
        if resultats_list:
            resultats = pd.DataFrame(resultats_list)
            # Afficher les résultats
            st.subheader("Résultats de l'analyse")
            for index, row in resultats.iterrows():
                st.markdown(f"### Rapport pour la vidéo {row['publication_id']}")
                st.markdown("**Questions les plus récurrentes ou similaires :**")
                st.write(row['questions_récurrentes'])
                st.markdown("**Principales questions non répondues :**")
                st.write(row['principales_questions'])
                st.markdown("---")

            # Option pour exporter les résultats
            st.subheader("Exportation des résultats")
            export_format = st.selectbox("Choisissez le format d'exportation :", options=['CSV', 'Excel'])
            if export_format == 'CSV':
                csv = resultats.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Télécharger les résultats en CSV",
                    data=csv,
                    file_name='resultats_analyse.csv',
                    mime='text/csv',
                )
            else:
                excel_file = 'resultats_analyse.xlsx'
                resultats.to_excel(excel_file, index=False)
                with open(excel_file, 'rb') as f:
                    st.download_button(
                        label="Télécharger les résultats en Excel",
                        data=f,
                        file_name='resultats_analyse.xlsx',
                        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                    )
        else:
            st.info("Aucun résultat à afficher. Veuillez vérifier vos paramètres d'analyse.")
else:
    st.info("Veuillez charger un fichier CSV pour commencer l'analyse.")

# Footer personnalisé
st.markdown("""
<hr>
<center>© 2023 Brut - Outil d'analyse des commentaires | Développé pour les journalistes de Brut</center>
""", unsafe_allow_html=True)
