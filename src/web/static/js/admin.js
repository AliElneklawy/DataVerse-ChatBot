/**
 * RAG Admin Dashboard JavaScript
 * Provides interactive functionality for the admin dashboard
 */

// Wait for DOM to be ready
document.addEventListener('DOMContentLoaded', function() {
    // Initialize tooltips
    const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function(tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Initialize popovers
    const popoverTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    popoverTriggerList.map(function(popoverTriggerEl) {
        return new bootstrap.Popover(popoverTriggerEl);
    });

    // Dark mode toggle
    const darkModeToggle = document.getElementById('darkModeToggle');
    if (darkModeToggle) {
        darkModeToggle.addEventListener('click', function() {
            document.body.classList.toggle('dark-mode');
            
            // Save preference in localStorage
            if (document.body.classList.contains('dark-mode')) {
                localStorage.setItem('darkMode', 'enabled');
            } else {
                localStorage.setItem('darkMode', 'disabled');
            }
        });
        
        // Check for saved preference
        if (localStorage.getItem('darkMode') === 'enabled') {
            document.body.classList.add('dark-mode');
            darkModeToggle.checked = true;
        }
    }

    // Sidebar toggle for mobile
    const sidebarToggle = document.getElementById('sidebarToggle');
    if (sidebarToggle) {
        sidebarToggle.addEventListener('click', function() {
            document.querySelector('.sidebar').classList.toggle('show');
        });
    }

    // Generic data table search functionality
    const tableSearchInputs = document.querySelectorAll('.table-search-input');
    tableSearchInputs.forEach(function(input) {
        input.addEventListener('keyup', function() {
            const tableId = this.getAttribute('data-table-id');
            const table = document.getElementById(tableId);
            if (!table) return;
            
            const term = this.value.toLowerCase();
            const rows = table.querySelectorAll('tbody tr');
            
            rows.forEach(function(row) {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(term) ? '' : 'none';
            });
        });
    });

    // Ajax form submission with loading state
    const ajaxForms = document.querySelectorAll('.ajax-form');
    ajaxForms.forEach(function(form) {
        form.addEventListener('submit', function(e) {
            e.preventDefault();
            
            const submitBtn = form.querySelector('[type="submit"]');
            const originalText = submitBtn.innerHTML;
            submitBtn.disabled = true;
            submitBtn.innerHTML = '<span class="spinner-border spinner-border-sm" role="status" aria-hidden="true"></span> Processing...';
            
            const formData = new FormData(form);
            
            fetch(form.action, {
                method: form.method,
                body: formData,
                headers: {
                    'X-Requested-With': 'XMLHttpRequest'
                }
            })
            .then(response => response.json())
            .then(data => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
                
                if (data.success) {
                    showNotification('Success', data.message || 'Operation completed successfully', 'success');
                    
                    // Handle redirect if specified
                    if (data.redirect) {
                        window.location.href = data.redirect;
                        return;
                    }
                    
                    // Handle form reset if specified
                    if (data.resetForm) {
                        form.reset();
                    }
                    
                    // Handle callback if specified
                    if (data.callback && typeof window[data.callback] === 'function') {
                        window[data.callback](data);
                    }
                } else {
                    showNotification('Error', data.message || 'An error occurred', 'danger');
                }
            })
            .catch(error => {
                submitBtn.disabled = false;
                submitBtn.innerHTML = originalText;
                showNotification('Error', 'An unexpected error occurred', 'danger');
                console.error('Form submission error:', error);
            });
        });
    });

    // Global notification system
    window.showNotification = function(title, message, type = 'info', duration = 5000) {
        const container = document.getElementById('notificationContainer');
        if (!container) {
            // Create container if it doesn't exist
            const newContainer = document.createElement('div');
            newContainer.id = 'notificationContainer';
            newContainer.style.position = 'fixed';
            newContainer.style.top = '10px';
            newContainer.style.right = '10px';
            newContainer.style.zIndex = '9999';
            document.body.appendChild(newContainer);
        }
        
        const notifId = 'notif-' + Date.now();
        const html = `
            <div id="${notifId}" class="toast" role="alert" aria-live="assertive" aria-atomic="true">
                <div class="toast-header bg-${type} text-white">
                    <strong class="me-auto">${title}</strong>
                    <button type="button" class="btn-close btn-close-white" data-bs-dismiss="toast" aria-label="Close"></button>
                </div>
                <div class="toast-body">
                    ${message}
                </div>
            </div>
        `;
        
        document.getElementById('notificationContainer').insertAdjacentHTML('beforeend', html);
        
        const toastEl = document.getElementById(notifId);
        const toast = new bootstrap.Toast(toastEl, { autohide: true, delay: duration });
        toast.show();
        
        // Remove from DOM after hiding
        toastEl.addEventListener('hidden.bs.toast', function() {
            toastEl.remove();
        });
    };

    // Confirmation dialogs
    const confirmButtons = document.querySelectorAll('[data-confirm]');
    confirmButtons.forEach(function(button) {
        button.addEventListener('click', function(e) {
            if (!confirm(this.getAttribute('data-confirm') || 'Are you sure?')) {
                e.preventDefault();
                e.stopPropagation();
            }
        });
    });

    // AJAX content loading
    const ajaxLoaders = document.querySelectorAll('[data-ajax-load]');
    ajaxLoaders.forEach(function(element) {
        element.addEventListener('click', function(e) {
            e.preventDefault();
            
            const targetId = this.getAttribute('data-target');
            const url = this.getAttribute('data-ajax-load');
            const targetElement = document.getElementById(targetId);
            
            if (!targetElement) return;
            
            targetElement.innerHTML = '<div class="text-center p-5"><div class="spinner-border" role="status"></div><p class="mt-3">Loading...</p></div>';
            
            fetch(url)
                .then(response => response.text())
                .then(html => {
                    targetElement.innerHTML = html;
                    // Initialize any new components
                    initializeComponents(targetElement);
                })
                .catch(error => {
                    targetElement.innerHTML = '<div class="alert alert-danger">Error loading content</div>';
                    console.error('AJAX loading error:', error);
                });
        });
    });

    // Initialize components in a specific container
    function initializeComponents(container) {
        // Re-initialize tooltips in the container
        const tooltips = container.querySelectorAll('[data-bs-toggle="tooltip"]');
        tooltips.forEach(el => new bootstrap.Tooltip(el));
        
        // Re-initialize popovers in the container
        const popovers = container.querySelectorAll('[data-bs-toggle="popover"]');
        popovers.forEach(el => new bootstrap.Popover(el));
    }

    // Copy to clipboard functionality
    const copyButtons = document.querySelectorAll('.btn-copy');
    copyButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            const textToCopy = this.getAttribute('data-copy-text');
            const targetId = this.getAttribute('data-copy-target');
            
            let textContent = textToCopy;
            if (!textContent && targetId) {
                const targetElement = document.getElementById(targetId);
                if (targetElement) {
                    textContent = targetElement.textContent || targetElement.value;
                }
            }
            
            if (textContent) {
                navigator.clipboard.writeText(textContent)
                    .then(() => {
                        // Show success indicator
                        const originalHtml = this.innerHTML;
                        this.innerHTML = '<i class="bi bi-check"></i>';
                        setTimeout(() => {
                            this.innerHTML = originalHtml;
                        }, 1500);
                    })
                    .catch(err => {
                        console.error('Copy failed:', err);
                        showNotification('Error', 'Failed to copy to clipboard', 'danger');
                    });
            }
        });
    });

    // Live data refresh for dashboards
    const liveElements = document.querySelectorAll('[data-refresh-url]');
    liveElements.forEach(function(element) {
        const url = element.getAttribute('data-refresh-url');
        const interval = parseInt(element.getAttribute('data-refresh-interval') || '30000', 10);
        
        if (url && interval > 0) {
            setInterval(function() {
                fetch(url)
                    .then(response => response.json())
                    .then(data => {
                        // Handle different element types
                        if (element.tagName === 'CANVAS') {
                            // Update chart data
                            const chartId = element.id;
                            const chartInstance = Chart.getChart(chartId);
                            if (chartInstance) {
                                // Update chart data based on structure
                                if (data.labels && data.datasets) {
                                    chartInstance.data.labels = data.labels;
                                    chartInstance.data.datasets = data.datasets;
                                    chartInstance.update();
                                }
                            }
                        } else {
                            // Update element content
                            const updateProperty = element.getAttribute('data-update-property') || 'textContent';
                            const updateKey = element.getAttribute('data-update-key');
                            
                            if (updateKey && updateKey in data) {
                                element[updateProperty] = data[updateKey];
                            } else {
                                // If no specific key, use the entire data as JSON string
                                element[updateProperty] = JSON.stringify(data);
                            }
                        }
                    })
                    .catch(error => {
                        console.error('Live refresh error:', error);
                    });
            }, interval);
        }
    });

    // Expandable code/content blocks
    const expandButtons = document.querySelectorAll('.btn-expand');
    expandButtons.forEach(function(button) {
        button.addEventListener('click', function() {
            const targetId = this.getAttribute('data-target');
            const target = document.getElementById(targetId);
            
            if (target) {
                target.classList.toggle('expanded');
                this.innerHTML = target.classList.contains('expanded') 
                    ? '<i class="bi bi-chevron-up"></i>' 
                    : '<i class="bi bi-chevron-down"></i>';
            }
        });
    });
});