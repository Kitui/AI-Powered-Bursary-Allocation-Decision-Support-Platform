(function () {
    function ready(fn) {
        if (document.readyState !== 'loading') {
            fn();
        } else {
            document.addEventListener('DOMContentLoaded', fn);
        }
    }

    ready(function () {
        var sidebar = document.querySelector('.sidebar');
        if (sidebar && !document.querySelector('.sidebar-toggle')) {
            var toggle = document.createElement('button');
            toggle.type = 'button';
            toggle.className = 'sidebar-toggle';
            toggle.setAttribute('aria-label', 'Toggle navigation');
            toggle.innerHTML = '<i class="fas fa-bars"></i>';
            document.body.appendChild(toggle);

            toggle.addEventListener('click', function () {
                if (window.matchMedia('(max-width: 900px)').matches) {
                    document.body.classList.toggle('sidebar-open');
                } else {
                    document.body.classList.toggle('sidebar-collapsed');
                }
            });
        }

        document.querySelectorAll('.card, .form-card, .stat-card, .chart-card, .task-card, .table-wrap, .filter-section').forEach(function (el, index) {
            el.classList.add('app-reveal');
            window.setTimeout(function () {
                el.classList.add('is-visible');
            }, Math.min(index * 45, 360));
        });

        document.querySelectorAll('input[type="password"]').forEach(function (input) {
            if (input.parentElement && input.parentElement.classList.contains('input-with-icon')) {
                var button = document.createElement('button');
                button.type = 'button';
                button.className = 'password-toggle';
                button.setAttribute('aria-label', 'Show password');
                button.innerHTML = '<i class="fas fa-eye"></i>';
                input.parentElement.appendChild(button);
                button.addEventListener('click', function () {
                    var isPassword = input.type === 'password';
                    input.type = isPassword ? 'text' : 'password';
                    button.innerHTML = isPassword ? '<i class="fas fa-eye-slash"></i>' : '<i class="fas fa-eye"></i>';
                });
            }
        });

        document.querySelectorAll('input[type="file"]').forEach(function (input) {
            input.addEventListener('change', function () {
                var label = input.closest('.form-group');
                var fileName = input.files && input.files[0] ? input.files[0].name : '';
                if (!label || !fileName) return;
                var note = label.querySelector('.file-selected-note');
                if (!note) {
                    note = document.createElement('div');
                    note.className = 'file-selected-note mt-2 text-muted';
                    label.appendChild(note);
                }
                note.innerHTML = '<i class="fas fa-file-csv"></i> ' + fileName;
            });
        });

        document.querySelectorAll('.btn').forEach(function (button) {
            button.addEventListener('mousedown', function () {
                button.style.transform = 'translateY(1px)';
            });
            button.addEventListener('mouseup', function () {
                button.style.transform = '';
            });
            button.addEventListener('mouseleave', function () {
                button.style.transform = '';
            });
        });
    });
})();
