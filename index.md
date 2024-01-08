{% for post in site.posts %}
##{{ post.title }} [link]({{ post.permalink }})
{{ post.excerpt}}
{% endfor %}
