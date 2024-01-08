{% for post in site.posts %}
[{{ post.permalink }}](##{{ post.title }})
{{ post.excerpt}}
{% endfor %}
